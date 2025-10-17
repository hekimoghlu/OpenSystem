/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//== RangedConstraintManager.h ----------------------------------*- C++ -*--==//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//
//
//  Ranged constraint manager, built on SimpleConstraintManager.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_STATICANALYZER_CORE_PATHSENSITIVE_RANGEDCONSTRAINTMANAGER_H
#define LANGUAGE_CORE_STATICANALYZER_CORE_PATHSENSITIVE_RANGEDCONSTRAINTMANAGER_H

#include "language/Core/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "language/Core/StaticAnalyzer/Core/PathSensitive/SimpleConstraintManager.h"
#include "toolchain/ADT/APSInt.h"
#include "toolchain/Support/Allocator.h"

namespace language::Core {

namespace ento {

/// A Range represents the closed range [from, to].  The caller must
/// guarantee that from <= to.  Note that Range is immutable, so as not
/// to subvert RangeSet's immutability.
class Range {
public:
  Range(const toolchain::APSInt &From, const toolchain::APSInt &To) : Impl(&From, &To) {
    assert(From <= To);
  }

  Range(const toolchain::APSInt &Point) : Range(Point, Point) {}

  bool Includes(const toolchain::APSInt &Point) const {
    return From() <= Point && Point <= To();
  }
  const toolchain::APSInt &From() const { return *Impl.first; }
  const toolchain::APSInt &To() const { return *Impl.second; }
  const toolchain::APSInt *getConcreteValue() const {
    return &From() == &To() ? &From() : nullptr;
  }

  void Profile(toolchain::FoldingSetNodeID &ID) const {
    ID.AddPointer(&From());
    ID.AddPointer(&To());
  }
  void dump(raw_ostream &OS) const;
  void dump() const;

  // In order to keep non-overlapping ranges sorted, we can compare only From
  // points.
  bool operator<(const Range &RHS) const { return From() < RHS.From(); }

  bool operator==(const Range &RHS) const { return Impl == RHS.Impl; }
  bool operator!=(const Range &RHS) const { return !operator==(RHS); }

private:
  std::pair<const toolchain::APSInt *, const toolchain::APSInt *> Impl;
};

/// @class RangeSet is a persistent set of non-overlapping ranges.
///
/// New RangeSet objects can be ONLY produced by RangeSet::Factory object, which
/// also supports the most common operations performed on range sets.
///
/// Empty set corresponds to an overly constrained symbol meaning that there
/// are no possible values for that symbol.
class RangeSet {
public:
  class Factory;

private:
  // We use toolchain::SmallVector as the underlying container for the following
  // reasons:
  //
  //   * Range sets are usually very simple, 1 or 2 ranges.
  //     That's why toolchain::ImmutableSet is not perfect.
  //
  //   * Ranges in sets are NOT overlapping, so it is natural to keep them
  //     sorted for efficient operations and queries.  For this reason,
  //     toolchain::SmallSet doesn't fit the requirements, it is not sorted when it
  //     is a vector.
  //
  //   * Range set operations usually a bit harder than add/remove a range.
  //     Complex operations might do many of those for just one range set.
  //     Formerly it used to be toolchain::ImmutableSet, which is inefficient for our
  //     purposes as we want to make these operations BOTH immutable AND
  //     efficient.
  //
  //   * Iteration over ranges is widespread and a more cache-friendly
  //     structure is preferred.
  using ImplType = toolchain::SmallVector<Range, 4>;

  struct ContainerType : public ImplType, public toolchain::FoldingSetNode {
    void Profile(toolchain::FoldingSetNodeID &ID) const {
      for (const Range &It : *this) {
        It.Profile(ID);
      }
    }
  };
  // This is a non-owning pointer to an actual container.
  // The memory is fully managed by the factory and is alive as long as the
  // factory itself is alive.
  // It is a pointer as opposed to a reference, so we can easily reassign
  // RangeSet objects.
  using UnderlyingType = const ContainerType *;
  UnderlyingType Impl;

public:
  using const_iterator = ImplType::const_iterator;

  const_iterator begin() const { return Impl->begin(); }
  const_iterator end() const { return Impl->end(); }
  size_t size() const { return Impl->size(); }

  bool isEmpty() const { return Impl->empty(); }

  class Factory {
  public:
    Factory(BasicValueFactory &BV) : ValueFactory(BV) {}

    /// Create a new set with all ranges from both LHS and RHS.
    /// Possible intersections are not checked here.
    ///
    /// Complexity: O(N + M)
    ///             where N = size(LHS), M = size(RHS)
    RangeSet add(RangeSet LHS, RangeSet RHS);
    /// Create a new set with all ranges from the original set plus the new one.
    /// Possible intersections are not checked here.
    ///
    /// Complexity: O(N)
    ///             where N = size(Original)
    RangeSet add(RangeSet Original, Range Element);
    /// Create a new set with all ranges from the original set plus the point.
    /// Possible intersections are not checked here.
    ///
    /// Complexity: O(N)
    ///             where N = size(Original)
    RangeSet add(RangeSet Original, const toolchain::APSInt &Point);
    /// Create a new set which is a union of two given ranges.
    /// Possible intersections are not checked here.
    ///
    /// Complexity: O(N + M)
    ///             where N = size(LHS), M = size(RHS)
    RangeSet unite(RangeSet LHS, RangeSet RHS);
    /// Create a new set by uniting given range set with the given range.
    /// All intersections and adjacent ranges are handled here.
    ///
    /// Complexity: O(N)
    ///             where N = size(Original)
    RangeSet unite(RangeSet Original, Range Element);
    /// Create a new set by uniting given range set with the given point.
    /// All intersections and adjacent ranges are handled here.
    ///
    /// Complexity: O(N)
    ///             where N = size(Original)
    RangeSet unite(RangeSet Original, toolchain::APSInt Point);
    /// Create a new set by uniting given range set with the given range
    /// between points. All intersections and adjacent ranges are handled here.
    ///
    /// Complexity: O(N)
    ///             where N = size(Original)
    RangeSet unite(RangeSet Original, toolchain::APSInt From, toolchain::APSInt To);

    RangeSet getEmptySet() { return &EmptySet; }

    /// Create a new set with just one range.
    /// @{
    RangeSet getRangeSet(Range Origin);
    RangeSet getRangeSet(const toolchain::APSInt &From, const toolchain::APSInt &To) {
      return getRangeSet(Range(From, To));
    }
    RangeSet getRangeSet(const toolchain::APSInt &Origin) {
      return getRangeSet(Origin, Origin);
    }
    /// @}

    /// Intersect the given range sets.
    ///
    /// Complexity: O(N + M)
    ///             where N = size(LHS), M = size(RHS)
    RangeSet intersect(RangeSet LHS, RangeSet RHS);
    /// Intersect the given set with the closed range [Lower, Upper].
    ///
    /// Unlike the Range type, this range uses modular arithmetic, corresponding
    /// to the common treatment of C integer overflow. Thus, if the Lower bound
    /// is greater than the Upper bound, the range is taken to wrap around. This
    /// is equivalent to taking the intersection with the two ranges [Min,
    /// Upper] and [Lower, Max], or, alternatively, /removing/ all integers
    /// between Upper and Lower.
    ///
    /// Complexity: O(N)
    ///             where N = size(What)
    RangeSet intersect(RangeSet What, toolchain::APSInt Lower, toolchain::APSInt Upper);
    /// Intersect the given range with the given point.
    ///
    /// The result can be either an empty set or a set containing the given
    /// point depending on whether the point is in the range set.
    ///
    /// Complexity: O(logN)
    ///             where N = size(What)
    RangeSet intersect(RangeSet What, toolchain::APSInt Point);

    /// Delete the given point from the range set.
    ///
    /// Complexity: O(N)
    ///             where N = size(From)
    RangeSet deletePoint(RangeSet From, const toolchain::APSInt &Point);
    /// Negate the given range set.
    ///
    /// Turn all [A, B] ranges to [-B, -A], when "-" is a C-like unary minus
    /// operation under the values of the type.
    ///
    /// We also handle MIN because applying unary minus to MIN does not change
    /// it.
    /// Example 1:
    /// char x = -128;        // -128 is a MIN value in a range of 'char'
    /// char y = -x;          // y: -128
    ///
    /// Example 2:
    /// unsigned char x = 0;  // 0 is a MIN value in a range of 'unsigned char'
    /// unsigned char y = -x; // y: 0
    ///
    /// And it makes us to separate the range
    /// like [MIN, N] to [MIN, MIN] U [-N, MAX].
    /// For instance, whole range is {-128..127} and subrange is [-128,-126],
    /// thus [-128,-127,-126,...] negates to [-128,...,126,127].
    ///
    /// Negate restores disrupted ranges on bounds,
    /// e.g. [MIN, B] => [MIN, MIN] U [-B, MAX] => [MIN, B].
    ///
    /// Negate is a self-inverse function, i.e. negate(negate(R)) == R.
    ///
    /// Complexity: O(N)
    ///             where N = size(What)
    RangeSet negate(RangeSet What);
    /// Performs promotions, truncations and conversions of the given set.
    ///
    /// This function is optimized for each of the six cast cases:
    /// - noop
    /// - conversion
    /// - truncation
    /// - truncation-conversion
    /// - promotion
    /// - promotion-conversion
    ///
    /// NOTE: This function is NOT self-inverse for truncations, because of
    ///       the higher bits loss:
    ///     - castTo(castTo(OrigRangeOfInt, char), int) != OrigRangeOfInt.
    ///     - castTo(castTo(OrigRangeOfChar, int), char) == OrigRangeOfChar.
    ///       But it is self-inverse for all the rest casts.
    ///
    /// Complexity:
    ///     - Noop                               O(1);
    ///     - Truncation                         O(N^2);
    ///     - Another case                       O(N);
    ///     where N = size(What)
    RangeSet castTo(RangeSet What, APSIntType Ty);
    RangeSet castTo(RangeSet What, QualType T);

    /// Return associated value factory.
    BasicValueFactory &getValueFactory() const { return ValueFactory; }

  private:
    /// Return a persistent version of the given container.
    RangeSet makePersistent(ContainerType &&From);
    /// Construct a new persistent version of the given container.
    ContainerType *construct(ContainerType &&From);

    RangeSet intersect(const ContainerType &LHS, const ContainerType &RHS);
    /// NOTE: This function relies on the fact that all values in the
    /// containers are persistent (created via BasicValueFactory::getValue).
    ContainerType unite(const ContainerType &LHS, const ContainerType &RHS);

    /// This is a helper function for `castTo` method. Implies not to be used
    /// separately.
    /// Performs a truncation case of a cast operation.
    ContainerType truncateTo(RangeSet What, APSIntType Ty);

    /// This is a helper function for `castTo` method. Implies not to be used
    /// separately.
    /// Performs a conversion case and a promotion-conversion case for signeds
    /// of a cast operation.
    ContainerType convertTo(RangeSet What, APSIntType Ty);

    /// This is a helper function for `castTo` method. Implies not to be used
    /// separately.
    /// Performs a promotion for unsigneds only.
    ContainerType promoteTo(RangeSet What, APSIntType Ty);

    // Many operations include producing new APSInt values and that's why
    // we need this factory.
    BasicValueFactory &ValueFactory;
    // Allocator for all the created containers.
    // Containers might own their own memory and that's why it is specific
    // for the type, so it calls container destructors upon deletion.
    toolchain::SpecificBumpPtrAllocator<ContainerType> Arena;
    // Usually we deal with the same ranges and range sets over and over.
    // Here we track all created containers and try not to repeat ourselves.
    toolchain::FoldingSet<ContainerType> Cache;
    static ContainerType EmptySet;
  };

  RangeSet(const RangeSet &) = default;
  RangeSet &operator=(const RangeSet &) = default;
  RangeSet(RangeSet &&) = default;
  RangeSet &operator=(RangeSet &&) = default;
  ~RangeSet() = default;

  /// Construct a new RangeSet representing '{ [From, To] }'.
  RangeSet(Factory &F, const toolchain::APSInt &From, const toolchain::APSInt &To)
      : RangeSet(F.getRangeSet(From, To)) {}

  /// Construct a new RangeSet representing the given point as a range.
  RangeSet(Factory &F, const toolchain::APSInt &Point)
      : RangeSet(F.getRangeSet(Point)) {}

  static void Profile(toolchain::FoldingSetNodeID &ID, const RangeSet &RS) {
    ID.AddPointer(RS.Impl);
  }

  /// Profile - Generates a hash profile of this RangeSet for use
  ///  by FoldingSet.
  void Profile(toolchain::FoldingSetNodeID &ID) const { Profile(ID, *this); }

  /// getConcreteValue - If a symbol is constrained to equal a specific integer
  ///  constant then this method returns that value.  Otherwise, it returns
  ///  NULL.
  const toolchain::APSInt *getConcreteValue() const {
    return Impl->size() == 1 ? begin()->getConcreteValue() : nullptr;
  }

  /// Get the minimal value covered by the ranges in the set.
  ///
  /// Complexity: O(1)
  const toolchain::APSInt &getMinValue() const;
  /// Get the maximal value covered by the ranges in the set.
  ///
  /// Complexity: O(1)
  const toolchain::APSInt &getMaxValue() const;

  bool isUnsigned() const;
  uint32_t getBitWidth() const;
  APSIntType getAPSIntType() const;

  /// Test whether the given point is contained by any of the ranges.
  ///
  /// Complexity: O(logN)
  ///             where N = size(this)
  bool contains(toolchain::APSInt Point) const { return containsImpl(Point); }

  bool containsZero() const {
    APSIntType T{getMinValue()};
    return contains(T.getZeroValue());
  }

  /// Test if the range is the [0,0] range.
  ///
  /// Complexity: O(1)
  bool encodesFalseRange() const {
    const toolchain::APSInt *Constant = getConcreteValue();
    return Constant && Constant->isZero();
  }

  /// Test if the range doesn't contain zero.
  ///
  /// Complexity: O(logN)
  ///             where N = size(this)
  bool encodesTrueRange() const { return !containsZero(); }

  void dump(raw_ostream &OS) const;
  void dump() const;

  bool operator==(const RangeSet &Other) const { return *Impl == *Other.Impl; }
  bool operator!=(const RangeSet &Other) const { return !(*this == Other); }

private:
  /* implicit */ RangeSet(ContainerType *RawContainer) : Impl(RawContainer) {}
  /* implicit */ RangeSet(UnderlyingType Ptr) : Impl(Ptr) {}

  /// Pin given points to the type represented by the current range set.
  ///
  /// This makes parameter points to be in-out parameters.
  /// In order to maintain consistent types across all of the ranges in the set
  /// and to keep all the operations to compare ONLY points of the same type, we
  /// need to pin every point before any operation.
  ///
  /// @Returns true if the given points can be converted to the target type
  ///          without changing the values (i.e. trivially) and false otherwise.
  /// @{
  bool pin(toolchain::APSInt &Lower, toolchain::APSInt &Upper) const;
  bool pin(toolchain::APSInt &Point) const;
  /// @}

  // This version of this function modifies its arguments (pins it).
  bool containsImpl(toolchain::APSInt &Point) const;

  friend class Factory;
};

using ConstraintMap = toolchain::ImmutableMap<SymbolRef, RangeSet>;
ConstraintMap getConstraintMap(ProgramStateRef State);

class RangedConstraintManager : public SimpleConstraintManager {
public:
  RangedConstraintManager(ExprEngine *EE, SValBuilder &SB)
      : SimpleConstraintManager(EE, SB) {}

  ~RangedConstraintManager() override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from SimpleConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSym(ProgramStateRef State, SymbolRef Sym,
                            bool Assumption) override;

  ProgramStateRef assumeSymInclusiveRange(ProgramStateRef State, SymbolRef Sym,
                                          const toolchain::APSInt &From,
                                          const toolchain::APSInt &To,
                                          bool InRange) override;

  ProgramStateRef assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                       bool Assumption) override;

protected:
  /// Assume a constraint between a symbolic expression and a concrete integer.
  virtual ProgramStateRef assumeSymRel(ProgramStateRef State, SymbolRef Sym,
                                       BinaryOperator::Opcode op,
                                       const toolchain::APSInt &Int);

  //===------------------------------------------------------------------===//
  // Interface that subclasses must implement.
  //===------------------------------------------------------------------===//

  // Each of these is of the form "$Sym+Adj <> V", where "<>" is the comparison
  // operation for the method being invoked.

  virtual ProgramStateRef assumeSymNE(ProgramStateRef State, SymbolRef Sym,
                                      const toolchain::APSInt &V,
                                      const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymEQ(ProgramStateRef State, SymbolRef Sym,
                                      const toolchain::APSInt &V,
                                      const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymLT(ProgramStateRef State, SymbolRef Sym,
                                      const toolchain::APSInt &V,
                                      const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymGT(ProgramStateRef State, SymbolRef Sym,
                                      const toolchain::APSInt &V,
                                      const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymLE(ProgramStateRef State, SymbolRef Sym,
                                      const toolchain::APSInt &V,
                                      const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymGE(ProgramStateRef State, SymbolRef Sym,
                                      const toolchain::APSInt &V,
                                      const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymWithinInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const toolchain::APSInt &From,
      const toolchain::APSInt &To, const toolchain::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymOutsideInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const toolchain::APSInt &From,
      const toolchain::APSInt &To, const toolchain::APSInt &Adjustment) = 0;

  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//
private:
  static void computeAdjustment(SymbolRef &Sym, toolchain::APSInt &Adjustment);
};

/// Try to simplify a given symbolic expression based on the constraints in
/// State. This is needed because the Environment bindings are not getting
/// updated when a new constraint is added to the State. If the symbol is
/// simplified to a non-symbol (e.g. to a constant) then the original symbol
/// is returned. We use this function in the family of assumeSymNE/EQ/LT/../GE
/// functions where we can work only with symbols. Use the other function
/// (simplifyToSVal) if you are interested in a simplification that may yield
/// a concrete constant value.
SymbolRef simplify(ProgramStateRef State, SymbolRef Sym);

/// Try to simplify a given symbolic expression's associated `SVal` based on the
/// constraints in State. This is very similar to `simplify`, but this function
/// always returns the simplified SVal. The simplified SVal might be a single
/// constant (i.e. `ConcreteInt`).
SVal simplifyToSVal(ProgramStateRef State, SymbolRef Sym);

} // namespace ento
} // namespace language::Core

REGISTER_FACTORY_WITH_PROGRAMSTATE(ConstraintMap)

#endif
