#include <tvm/arith/analyzer.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/type.h>
#include <tvm/relay/op.h>
#include <tvm/tir/function.h>

#include "../../ir/scope.h"

namespace tvm {
namespace relax {

using tvm::ir::BaseScope;
using tvm::ir::BaseScopeNode;
using tvm::ir::ScopeManagerNode;

class IRBuilderNode : public ScopeManagerNode {
 protected:
  using ScopeManagerNode::GetScope;
  using ScopeManagerNode::PopScope;
  using ScopeManagerNode::PushScope;
  // using ScopeManagerNode::name2value;
  // using ScopeManagerNode::scopes;

 public:
  Map<Id, Expr> id2bind;

  void VisitAttrs(AttrVisitor* v) {
    ScopeManagerNode::VisitAttrs(v);
    v->Visit("id2bind", &id2bind);
  }
  static constexpr const char* _type_key = "relax.IRBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRBuilderNode, ScopeManagerNode);

 public:
  // Utility methods
  bool CurrentBlockIsDataFlow() const;
  Expr LookupBinding(const Id& id) const;
  // Scope push/pop
  void BeginDataflowBlock();
  void BeginBindingBlock();
  BindingBlock EndBlock();
  // Manipulating the outmost IRModule scope
  GlobalVar AddFuncToContext(const BaseFunc& func, const String& func_name_hint);
  IRModule GetContextIRModule() const;
  // Emit bindings
  Var Emit(const VarBinding& binding);
  Var EmitMatchShape(const MatchShape& binding);
  Var EmitOutput(const VarBinding& binding);
  // Create and emit bindings
  Var Emit(const Expr& expr, const Optional<String>& name_hint = NullOpt);
  Var EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern,
                     const Optional<String>& name_hint = NullOpt);
  Var EmitOutput(const Expr& output, const Optional<String>& name_hint);
  // Misc
  bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs);

 protected:
  Expr Normalize(const Expr& expr);  // TBD
};

// DataflowScope

class DataflowScopeNode : public BaseScopeNode {
 public:
  Array<Binding> bindings;

  void VisitAttrs(tvm::AttrVisitor* v) {
    BaseScopeNode::VisitAttrs(v);
    v->Visit("bindings", &bindings);
  }

  static constexpr const char* _type_key = "relax.DataflowScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowScopeNode, BaseScopeNode);
};

class DataflowScope : public BaseScope {
 public:
  DataflowScope() { data_ = make_object<DataflowScopeNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(DataflowScope, BaseScope, DataflowScopeNode);
};

// NonDataflowScope

class NonDataflowScopeNode : public BaseScopeNode {
 public:
  Array<Binding> bindings;
  void VisitAttrs(tvm::AttrVisitor* v) {
    BaseScopeNode::VisitAttrs(v);
    v->Visit("bindings", &bindings);
  }
  static constexpr const char* _type_key = "relax.NonDataflowScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(NonDataflowScopeNode, BaseScopeNode);
};

class NonDataflowScope : public BaseScope {
 public:
  NonDataflowScope() { data_ = make_object<NonDataflowScopeNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(NonDataflowScope, BaseScope,
                                                    NonDataflowScopeNode);
};

TVM_REGISTER_NODE_TYPE(DataflowScopeNode);
TVM_REGISTER_NODE_TYPE(NonDataflowScopeNode);
TVM_REGISTER_NODE_TYPE(IRBuilderNode);

Optional<Expr> InferShape(const Call& call) {
  if (call->shape_.defined()) {
    return Downcast<Expr>(call->shape_.value());
  }
  static auto op_map = Op::GetAttrMap<FInferShape>("FInferShape");
  thread_local DiagnosticContext diag_ctx = DiagnosticContext::Default(IRModule({}, {}));
  if (const auto* op_node = call->op.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    if (op_map.count(op)) {
      return op_map[op](call, diag_ctx);
    }
  }
  return NullOpt;
}

Type InferType(const Call& call) {
  if (call->checked_type_.defined()) {
    return call->checked_type_;
  }
  static auto op_map = Op::GetAttrMap<FInferType>("FInferType");
  thread_local DiagnosticContext diag_ctx = DiagnosticContext::Default(IRModule({}, {}));
  if (const auto* op_node = call->op.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    if (op_map.count(op)) {
      return op_map[op](call, diag_ctx);
    }
  }
  return Type();
}

bool IRBuilderNode::CurrentBlockIsDataFlow() const {
  return scopes.back()->IsInstance<DataflowScopeNode>();
}

Expr IRBuilderNode::LookupBinding(const Id& id) const {
  auto it = id2bind.find(id);
  if (it == id2bind.end()) {
    LOG(FATAL) << "Cannot find binding for " << id;
  }
  return (*it).second;
}

void IRBuilderNode::BeginDataflowBlock() { scopes.push_back(DataflowScope()); }
void IRBuilderNode::BeginBindingBlock() { scopes.push_back(NonDataflowScope()); }

BindingBlock IRBuilderNode::EndBlock() {
  CHECK(!scopes.empty());
  BaseScope scope = this->PopScope();
  if (const auto* n = scope.as<DataflowScopeNode>()) {
    return DataflowBlock(n->bindings);
  }
  if (const auto* n = scope.as<NonDataflowScopeNode>()) {
    return BindingBlock(n->bindings);
  }
  ICHECK(false) << "Invalid scope: " << scope;
  throw;
}

GlobalVar IRBuilderNode::AddFuncToContext(const BaseFunc& func, const String& func_name_hint) {
  if (Optional<ir::IRModuleScope> scope = GetScope<ir::IRModuleScope>()) {
    return scope.value()->Add(func_name_hint, func);
  }
  ICHECK(false) << "No IRModuleScope in context";
  throw;
}

IRModule IRBuilderNode::GetContextIRModule() const {
  if (Optional<ir::IRModuleScope> scope = GetScope<ir::IRModuleScope>()) {
    return scope.value()->AsIRModule();
  }
  ICHECK(false) << "No IRModuleScope in context";
  throw;
}

Var IRBuilderNode::Emit(const VarBinding& binding) {
  BaseScope scope = scopes.back();
  if (auto* n = scope.as<DataflowScopeNode>()) {
    ICHECK(binding->var.as<DataflowVarNode>())
        << "Emit can only be used for local bindings in a dataflow block, use EmitOutput for "
           "output bindings instead";
    n->bindings.push_back(binding);
  } else if (auto* n = scope.as<NonDataflowScopeNode>()) {
    n->bindings.push_back(binding);
  } else {
    LOG(FATAL) << "Invalid scope: " << scope;
  }
  this->id2bind.Set(binding->var->vid, binding->value);
  return binding->var;
}

Var IRBuilderNode::EmitOutput(const VarBinding& binding) {
  BaseScope scope = scopes.back();
  if (auto* n = scope.as<DataflowScopeNode>()) {
    ICHECK(binding->var->IsInstance<DataflowVarNode>())
        << "EmitOutput has to be called inside dataflow block.";
    n->bindings.push_back(binding);
  } else {
    LOG(FATAL) << "Invalid scope: " << scope;
  }
  this->id2bind.Set(binding->var->vid, binding->value);
  return binding->var;
}

Var IRBuilderNode::EmitMatchShape(const MatchShape& binding) {
  BaseScope scope = scopes.back();
  if (auto* n = scope.as<DataflowScopeNode>()) {
    ICHECK(binding->var->IsInstance<DataflowVarNode>())
        << "EmitMatchShape can only be used for local bindings in a dataflow block.";
    n->bindings.push_back(binding);
  } else if (auto* n = scope.as<NonDataflowScopeNode>()) {
    ICHECK(!binding->var->IsInstance<DataflowVarNode>())
        << "cannot emit dataflow vars outside a dataflow block: " << binding->var;
    n->bindings.push_back(binding);
  } else {
    LOG(FATAL) << "Invalid scope: " << scope;
  }
  // TODO(@junrushao1994): id2bind?
  // this->id2bind.Set(binding->var->vid, binding->value);
  // TODO(@altanh, @yuchen): what value should we bind? Consider
  //    y = add(x, x)
  //    z = match_shape(y, (n, m))
  // We would like pass writers to match "z" with the "add" node but with extra shape info.
  // Maybe this logic could be deferred to a DFPattern-style rewriter?
  return binding->var;
}

inline Var CreateScopedVar(const IRBuilderNode* self, const Optional<String>& name_hint) {
  BaseScope scope = self->scopes.back();
  if (scope->IsInstance<DataflowScopeNode>()) {
    return DataflowVar(/*id=*/Id(self->GetUniqueName(name_hint.value_or("lv"))),
                       /*shape_annotation=*/NullOpt,  //
                       /*type_annotation=*/NullOpt);
  } else if (scope->IsInstance<NonDataflowScopeNode>()) {
    return Var(/*id=*/Id(self->GetUniqueName(name_hint.value_or("gv"))),
               /*shape_annotation=*/NullOpt,  //
               /*type_annotation=*/NullOpt);
  }
  LOG(FATAL) << "Invalid scope: " << scope;
  throw;
}

inline VarBinding CreateVarBinding(const Var& var, const Expr& expr) {
  if (const CallNode* call_node = expr.as<CallNode>()) {
    Call call = GetRef<Call>(call_node);
    Call new_call(make_object<CallNode>(*call_node));
    var->shape_ = new_call->shape_ = InferShape(call);
    var->checked_type_ = new_call->checked_type_ = InferType(call);
  } else if (const VarNode* rhs_var = expr.as<VarNode>()) {
    var->shape_ = rhs_var->shape_;
    var->checked_type_ = rhs_var->checked_type_;
  } else if (const TupleGetItemNode* tuple_get_item = expr.as<TupleGetItemNode>()) {
    const auto* rhs_var = tuple_get_item->tuple.as<VarNode>();
    ICHECK(rhs_var) << "TypeError: Invalid type as the tuple field of TupleGetItem: "
                    << tuple_get_item->tuple->GetTypeKey();
    int i = tuple_get_item->index;
    if (const TupleNode* shape = rhs_var->shape_.as<TupleNode>()) {
      var->shape_ = shape->fields[i];
    }
    if (const TupleTypeNode* type = rhs_var->checked_type_.as<TupleTypeNode>()) {
      var->checked_type_ = type->fields[i];
    }
  }
  return VarBinding(var, expr);
}

Var IRBuilderNode::Emit(const Expr& expr, const Optional<String>& name_hint) {
  return this->Emit(CreateVarBinding(CreateScopedVar(this, name_hint), expr));
}

Var IRBuilderNode::EmitMatchShape(const Expr& value, const Array<PrimExpr>& pattern,
                                  const Optional<String>& name_hint) {
  Var var = CreateScopedVar(this, name_hint);
  Type ty = value->checked_type();
  if (ty->IsInstance<ShapeTypeNode>()) {
    var->checked_type_ = ShapeType(Span());
  } else if (const DynTensorTypeNode* tty = ty.as<DynTensorTypeNode>()) {
    var->shape_ = ShapeExpr(pattern);
    var->checked_type_ = DynTensorType(pattern.size(), tty->dtype);
  } else {
    LOG(FATAL) << "TypeError: Invalid type of value, must be of DynTensorType or ShapeType: " << ty;
  }
  return this->EmitMatchShape(MatchShape(value, pattern, var));
}

Var IRBuilderNode::EmitOutput(const Expr& output, const Optional<String>& name_hint) {
  BaseScope scope = scopes.back();
  if (scope->IsInstance<DataflowScopeNode>()) {
    return this->Emit(CreateVarBinding(Var(/*id=*/Id(this->GetUniqueName(name_hint.value_or("gv"))),
                                           /*shape_annotation=*/NullOpt,
                                           /*type_annotation=*/NullOpt),
                                       output));
  }
  LOG(FATAL) << "Invalid scope: " << this->scopes.back();
  throw;
}

}  // namespace relax
}  // namespace tvm
