#pragma once

#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/types.h>

/* IR Simplification
 *
 * Simplfies expressions in two stages:
 *  1. Recursively traverse the map combining similar operations into Terms
 * (interacted via Multiplication) and Polynomials (interacted via Addition). We
 * reorder the components of each Term or Polynomial into a consistent order to
 * allow combination or cancelling of like terms.
 *  2. Once the format of the tree is minimal, expand each Term into a sequence
 * of Muls, and each Polynomial into a sequence of Ads.
 */

namespace torch {
namespace jit {
namespace tensorexpr {

// A bunch of helpers for determine the Dtype of the output of a multi argument
// Term or Polynomial.
namespace {
template <class ExprType>
Dtype promoteTypesVec(const Expr* s, std::vector<const ExprType*>& v) {
  Dtype t = s->dtype();
  bool first = true;

  for (auto* e : v) {
    if (first) {
      t = Dtype(t.scalar_type(), e->dtype().lanes());
      first = false;
    }
    t = promoteTypes(t, e->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesVec(std::vector<const ExprType*>& v) {
  if (v.empty()) {
    throw malformed_input("empty list of types");
  }

  Dtype t = v[0]->dtype();
  for (auto* e : v) {
    t = promoteTypes(t, e->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesMap(
    const Expr* s,
    std::unordered_map<SimplifierHashType, const ExprType*>& m) {
  Dtype t = s->dtype();
  bool first = true;
  for (auto& e : m) {
    if (first) {
      t = Dtype(t.scalar_type(), e.second->dtype().lanes());
      first = false;
    }
    t = promoteTypes(t, e.second->dtype());
  }
  return t;
}

template <class ExprType>
Dtype promoteTypesVar(const ExprType* e) {
  return e->dtype();
}

template <class ExprType, class... Args>
Dtype promoteTypesVar(const ExprType* e, Args... es) {
  Dtype lhs = e->dtype();
  Dtype rhs = promoteTypesVar(es...);
  if (e->isConstant()) {
    lhs = Dtype(lhs.scalar_type(), rhs.lanes());
  }

  return promoteTypes(lhs, rhs);
}

// Helper for determining if an Expr is a multi-lane primitive (e.g. Broadcast
// or Ramp).
bool isMultilanePrimitive(const Expr* e) {
  return e->expr_type() == IRNodeType::kBroadcast ||
      e->expr_type() == IRNodeType::kRamp;
}
} // namespace

// A Term represents a grouping of Exprs through multiplication.
// E.g. product(scalar, *variables).
class Term : public ExprNode<Term> {
 public:
  template <class... Args>
  Term(HashProvider& hasher, const Expr* s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addComponent(ts...);
    sort();
  }

  Term(HashProvider& hasher, const Expr* s, std::vector<const Expr*> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor from a map of hash -> var, used when merging Terms.
  Term(
      HashProvider& hasher,
      const Expr* s,
      std::unordered_map<SimplifierHashType, const Expr*> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addComponent(p.second);
    }
    sort();
  }

  const Expr* scalar() const {
    return scalar_;
  }
  const std::vector<const Expr*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  // Produce a hash of just the variable components of this term, to determine
  // if it can be combined with another term.
  SimplifierHashType hashVars() const;

 private:
  std::vector<const Expr*> variables_;
  const Expr* scalar_;
  HashProvider& hasher_;

  void addComponent() {}
  void addComponent(const Expr* e) {
    variables_.push_back(e);
  }
  template <class... Es>
  void addComponent(const Expr* e, Es... es) {
    addComponent(e);
    addComponent(es...);
  }

  // Sort by hash to normalize order of components.
  void sort();
};

// Polynomial represents a grouping of Exprs by addition.
// E.g. sum(*variables, scalar).
// This would better be called Expression, but, naming conflict...
class Polynomial : public ExprNode<Polynomial> {
 public:
  template <class... Args>
  Polynomial(HashProvider& hasher, const Expr* s, Args... ts)
      : ExprNodeBase(promoteTypesVar(s, ts...)), scalar_(s), hasher_(hasher) {
    CHECK(s->isConstant());
    addTerm(ts...);
    sort();
  }

  Polynomial(HashProvider& hasher, const Expr* s, std::vector<const Term*> v)
      : ExprNodeBase(promoteTypesVec(s, v)),
        variables_(std::move(v)),
        scalar_(s),
        hasher_(hasher) {
    sort();
  }

  // Helper constructor for list of terms with no scalar component.
  Polynomial(HashProvider& hasher, std::vector<const Term*> terms)
      : ExprNodeBase(promoteTypesVec(terms)),
        variables_(std::move(terms)),
        scalar_(getImmediateByType(dtype(), 0)),
        hasher_(hasher) {
    sort();
  }

  // Convenience constructor for map of hash -> var, used when merging
  // Polynomials.
  Polynomial(
      HashProvider& hasher,
      const Expr* s,
      std::unordered_map<SimplifierHashType, const Term*> varmap)
      : ExprNodeBase(promoteTypesMap(s, varmap)), scalar_(s), hasher_(hasher) {
    for (auto& p : varmap) {
      addTerm(p.second);
    }
    sort();
  }

  const Expr* scalar() const {
    return scalar_;
  }
  const std::vector<const Term*>& variables() const {
    return variables_;
  }
  HashProvider& hasher() const {
    return hasher_;
  }

  SimplifierHashType hashVars() const;

 private:
  std::vector<const Term*> variables_;
  const Expr* scalar_;
  HashProvider& hasher_;

  void addTerm(const Term* t) {
    variables_.push_back(t);
  }
  template <class... Ts>
  void addTerm(const Term* t, Ts... ts) {
    addTerm(t);
    addTerm(ts...);
  }

  // Sort by hash to normalize order of terms.
  void sort();
};

class RoundOff : public BinaryOpNode<RoundOff> {
 public:
  RoundOff(const Expr* lhs, const Expr* rhs)
      : BinaryOpNode(lhs, rhs, IRNodeType::kRoundOff) {}
};

// Simplify the IR by combining arithmetic expressions over common terms.
class TORCH_API PolynomialTransformer : public IRMutator {
 public:
  // Inserts term into the provided map, in the case of a hash collision
  // combines the term with the existing and updates the map.
  void addOrUpdateTerm(
      std::unordered_map<SimplifierHashType, const Term*>& varmap,
      const Term* term);

  // Add Polynomial expressions, combining Terms representing the same
  // variables.
  const Expr* addPolynomials(const Polynomial* lhs, const Polynomial* rhs);

  // Insert a new Term into the provided polynomial. If the new term has common
  // variables to an existing term it is combined.
  const Expr* insertTerm(const Polynomial* poly, const Term* term);

  // Merge and simplify addition.
  const Expr* mutate(const Add* v) override;

  // Subtract one term from another, cancelling if necessary.
  const Expr* subTerms(const Term* lhs, const Term* rhs, bool negated);

  // Subtract the RHS Polynomial from the LHS Polynomial, cancelling out where
  // possible.
  const Expr* subPolynomials(const Polynomial* lhs, const Polynomial* rhs);

  // Merge and simplify subtraction.
  const Expr* mutate(const Sub* v) override;

  // Multiply two terms together, usually creating a new term with the variable
  // lists concatenated.
  const Term* mulTerms(const Term* lhs, const Term* rhs);

  // Multiply a Polynomial by a Term.
  const Expr* polyByTerm(const Polynomial* poly, const Term* term);

  // Match a rounding pattern and create a RoundOff if found.
  const Expr* isRoundOff(const Expr* lhs, const Expr* rhs);

  // Inserts a new component into a term, simplifying if possible.
  const Expr* insertIntoTerm(const Term* term, const Expr* expr);

  // Merge and simplify multiplication.
  const Expr* mutate(const Mul* v) override;

  const Expr* mutate(const Div* v) override;

  const Expr* mutate(const Mod* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const And* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Xor* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Lshift* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Rshift* v) override {
    return mutateBinaryOp(v, this);
  }

  const Expr* mutate(const Max* v) override {
    return mutateBinaryOp(v, this, v->propagate_nans());
  }

  const Expr* mutate(const Min* v) override {
    return mutateBinaryOp(v, this, v->propagate_nans());
  }

  const Expr* mutate(const Intrinsics* v) override;

  const Expr* mutate(const Cast* v) override;

  template <typename Op>
  static const Expr* mutateBinaryOp(
      const BinaryOpNode<Op>* v,
      IRMutator* mutator,
      bool option = false) {
    const Expr* lhs = v->lhs();
    const Expr* rhs = v->rhs();
    const Expr* lhs_new = lhs->accept_mutator(mutator);
    const Expr* rhs_new = rhs->accept_mutator(mutator);

    const Expr* node = v;

    if (lhs != lhs_new || rhs != rhs_new) {
      node = newBinaryOpOfType(v->expr_type(), lhs_new, rhs_new, option);
    }

    // Can only fold if both sides are constant.
    if (!lhs_new->isConstant() || !rhs_new->isConstant()) {
      return node;
    }

    return evaluateOp(node);
  }

  HashProvider& hasher() {
    return hasher_;
  }

  static const Expr* simplify(const Expr* e);
  static ExprHandle simplify(const ExprHandle& e);
  static Stmt* simplify(Stmt* e);

 private:
  HashProvider hasher_;
}; // namespace tensorexpr

// Expands Terms and Polynomial expressions into primitive operations.
// Does some simple factorization and reordering.
class TORCH_API TermExpander : public IRMutator {
  PolynomialTransformer* simplifier_;

 public:
  TermExpander(PolynomialTransformer* simplifier) : simplifier_(simplifier) {}

  // Expand Terms out to a series of Muls.
  const Expr* mutate(const Term* v) override;

  // Trivially factorize terms by GCD of scalar components.
  const Expr* factorizePolynomial(const Polynomial* poly);

  // Expand Polynomials out to a series of Adds.
  const Expr* mutate(const Polynomial* v) override;

  // Expand RoundOff to it's component: Mul(Div(lhs, rhs), rhs).
  const Expr* mutate(const RoundOff* v) override;
};

class TORCH_API IRSimplifier {
 public:
  static const Expr* simplify(const Expr* e) {
    PolynomialTransformer simplifier;
    e = e->accept_mutator(&simplifier);

    // There may be terms left in the IR, expand them.
    TermExpander expander(&simplifier);
    e = e->accept_mutator(&expander);

    return e;
  }

  static ExprHandle simplify(const ExprHandle& e) {
    return ExprHandle(simplify(e.node()));
  }

  static Stmt* simplify(Stmt* s) {
    PolynomialTransformer simplifier;
    s = s->accept_mutator(&simplifier);

    // There may be terms left in the IR, expand them.
    TermExpander expander(&simplifier);
    s = s->accept_mutator(&expander);

    return s;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
