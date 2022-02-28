#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/node/node.h>

namespace tvm {
namespace ir {

// Callback

class CallbackNode : public runtime::Object {
 public:
  runtime::TypedPackedFunc<void()> fn;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // v->Visit("fn", &fn);
  }

  static constexpr const char* _type_key = "ir.CallbackNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallbackNode, runtime::Object);
};

class Callback : public runtime::ObjectRef {
 public:
  explicit Callback(runtime::TypedPackedFunc<void()> fn) {
    ObjectPtr<CallbackNode> node = make_object<CallbackNode>();
    node->fn = std::move(fn);
    data_ = std::move(node);
  }
  void operator()() const { (*this)->fn(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Callback, runtime::ObjectRef, CallbackNode);
};

// BaseScope

class BaseScopeNode : public runtime::Object {
 public:
  Array<String> vars;
  Array<Callback> callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("vars", &vars);
    v->Visit("callbacks", &callbacks);
  }
  virtual ~BaseScopeNode() = default;
  virtual void AddVar(const String& var) { vars.push_back(var); }
  virtual void AddCallback(const Callback& cb) { callbacks.push_back(cb); }

  static constexpr const char* _type_key = "ir.BaseScope";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseScopeNode, Object);
};

class BaseScope : public runtime::ObjectRef {
 public:
  BaseScope() { data_ = make_object<BaseScopeNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BaseScope, runtime::ObjectRef, BaseScopeNode);
  template <typename ObjectType>
  inline ObjectType* as() const {
    if (this->get()->template IsInstance<ObjectType>()) {
      return static_cast<ObjectType*>(this->get());
    } else {
      return nullptr;
    }
  }
};

// ScopeManager

class ScopeManagerNode : public runtime::Object {
 public:
  Array<BaseScope> scopes;
  Map<String, ObjectRef> name2value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("scopes", &scopes);
    v->Visit("name2value", &name2value);
  }

  String GetUniqueName(const String& prefix) const {
    for (int i = 0;; i++) {
      String name = prefix + std::to_string(i);
      if (name2value.count(name) == 0) {
        return name;
      }
    }
    throw;
  }

  template <class TScope>
  Optional<TScope> GetScope() const {
    using TScopeNode = typename TScope::ContainerType;
    int n = scopes.size();
    for (int i = n - 1; i >= 0; --i) {
      if (const auto* n = scopes[i].template as<TScopeNode>()) {
        return GetRef<TScope>(n);
      }
    }
    return NullOpt;
  }

  BaseScope PushScope(BaseScope scope) {
    scopes.push_back(scope);
    return scope;
  }

  BaseScope PopScope() {
    BaseScope scope = this->scopes.back();
    this->scopes.pop_back();
    for (const Callback& cb : scope->callbacks) {
      cb();
    }
    for (const String& name : scope->vars) {
      name2value.erase(name);
    }
    return scope;
  }

  static constexpr const char* _type_key = "ir.ScopeManager";
  TVM_DECLARE_BASE_OBJECT_INFO(ScopeManagerNode, runtime::Object);

 protected:
  Optional<ObjectRef> GetVar(const String& name) const {
    // TODO: use template
    auto it = name2value.find(name);
    if (it != name2value.end()) {
      return (*it).second;
    }
    return NullOpt;
  }

  void AddVars(const String& name, const ObjectRef& value, const BaseScope& scope) {
    if (name2value.count(name) != 0) {
      LOG(FATAL) << "Variable already exists: " << name;
    }
    name2value.Set(name, value);
    scope->AddVar(name);
  }
};

class ScopeManager : public runtime::ObjectRef {
 public:
  ScopeManager() { data_ = make_object<ScopeManagerNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ScopeManager, runtime::ObjectRef,
                                                    ScopeManagerNode);
};

// IRModuleScope

class IRModuleScopeNode : public BaseScopeNode {
 public:
  Map<GlobalVar, BaseFunc> func_map;
  std::unordered_map<BaseFunc, GlobalVar, StructuralHash, StructuralEqual> structural_map;

  GlobalVar Add(const String& name, const BaseFunc& func) {
    auto it = structural_map.find(func);
    if (it != structural_map.end()) {
      return it->second;
    }
    GlobalVar gv(name);
    structural_map[func] = gv;
    func_map.Set(gv, func);
    return gv;
  }

  IRModule AsIRModule() const { return IRModule(func_map); }

  void VisitAttrs(tvm::AttrVisitor* v) {
    BaseScopeNode::VisitAttrs(v);
    v->Visit("func_map", &func_map);
    // v->Visit("structural_map", &structural_map);
  }

  static constexpr const char* _type_key = "ir.IRModuleScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRModuleScopeNode, BaseScopeNode);
};

class IRModuleScope : public BaseScope {
 public:
  IRModuleScope() { data_ = make_object<IRModuleScopeNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRModuleScope, BaseScope, IRModuleScopeNode);
};

}  // namespace ir
}  // namespace tvm
