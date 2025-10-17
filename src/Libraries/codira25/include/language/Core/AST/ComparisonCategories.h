/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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

//===- ComparisonCategories.h - Three Way Comparison Data -------*- C++ -*-===//
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
//  This file defines the Comparison Category enum and data types, which
//  store the types and expressions needed to support operator<=>
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_AST_COMPARISONCATEGORIES_H
#define LANGUAGE_CORE_AST_COMPARISONCATEGORIES_H

#include "language/Core/Basic/LLVM.h"
#include "toolchain/ADT/APSInt.h"
#include "toolchain/ADT/DenseMap.h"
#include <array>
#include <cassert>
#include <optional>
#include <vector>

namespace toolchain {
  class StringRef;
  class APSInt;
}

namespace language::Core {

class ASTContext;
class VarDecl;
class CXXRecordDecl;
class Sema;
class QualType;
class NamespaceDecl;

/// An enumeration representing the different comparison categories
/// types.
///
/// C++20 [cmp.categories.pre] The types partial_ordering, weak_ordering, and
/// strong_ordering are collectively termed the comparison category types.
enum class ComparisonCategoryType : unsigned char {
  PartialOrdering,
  WeakOrdering,
  StrongOrdering,
  First = PartialOrdering,
  Last = StrongOrdering
};

/// Determine the common comparison type, as defined in C++2a
/// [class.spaceship]p4.
inline ComparisonCategoryType commonComparisonType(ComparisonCategoryType A,
                                                   ComparisonCategoryType B) {
  return A < B ? A : B;
}

/// Get the comparison category that should be used when comparing values of
/// type \c T.
std::optional<ComparisonCategoryType>
getComparisonCategoryForBuiltinCmp(QualType T);

/// An enumeration representing the possible results of a three-way
/// comparison. These values map onto instances of comparison category types
/// defined in the standard library. e.g. 'std::strong_ordering::less'.
enum class ComparisonCategoryResult : unsigned char {
  Equal,
  Equivalent,
  Less,
  Greater,
  Unordered,
  Last = Unordered
};

class ComparisonCategoryInfo {
  friend class ComparisonCategories;
  friend class Sema;

public:
  ComparisonCategoryInfo(const ASTContext &Ctx, const CXXRecordDecl *RD,
                         ComparisonCategoryType Kind)
      : Ctx(Ctx), Record(RD), Kind(Kind) {}

  struct ValueInfo {
    ComparisonCategoryResult Kind;
    VarDecl *VD;

    ValueInfo(ComparisonCategoryResult Kind, VarDecl *VD)
        : Kind(Kind), VD(VD) {}

    /// True iff we've successfully evaluated the variable as a constant
    /// expression and extracted its integer value.
    bool hasValidIntValue() const;

    /// Get the constant integer value used by this variable to represent
    /// the comparison category result type.
    toolchain::APSInt getIntValue() const;
  };
private:
  const ASTContext &Ctx;

  /// A map containing the comparison category result decls from the
  /// standard library. The key is a value of ComparisonCategoryResult.
  mutable toolchain::SmallVector<
      ValueInfo, static_cast<unsigned>(ComparisonCategoryResult::Last) + 1>
      Objects;

  /// Lookup the ValueInfo struct for the specified ValueKind. If the
  /// VarDecl for the value cannot be found, nullptr is returned.
  ///
  /// If the ValueInfo does not have a valid integer value the variable
  /// is evaluated as a constant expression to determine that value.
  ValueInfo *lookupValueInfo(ComparisonCategoryResult ValueKind) const;

public:
  /// The declaration for the comparison category type from the
  /// standard library.
  const CXXRecordDecl *Record = nullptr;

  /// The Kind of the comparison category type
  ComparisonCategoryType Kind;

public:
  QualType getType() const;

  const ValueInfo *getValueInfo(ComparisonCategoryResult ValueKind) const {
    ValueInfo *Info = lookupValueInfo(ValueKind);
    assert(Info &&
           "comparison category does not contain the specified result kind");
    assert(Info->hasValidIntValue() &&
           "couldn't determine the integer constant for this value");
    return Info;
  }

  /// True iff the comparison is "strong". i.e. it checks equality and
  /// not equivalence.
  bool isStrong() const {
    using CCK = ComparisonCategoryType;
    return Kind == CCK::StrongOrdering;
  }

  /// True iff the comparison is not totally ordered.
  bool isPartial() const {
    using CCK = ComparisonCategoryType;
    return Kind == CCK::PartialOrdering;
  }

  /// Converts the specified result kind into the correct result kind
  /// for this category. Specifically it lowers strong equality results to
  /// weak equivalence if needed.
  ComparisonCategoryResult makeWeakResult(ComparisonCategoryResult Res) const {
    using CCR = ComparisonCategoryResult;
    if (!isStrong() && Res == CCR::Equal)
      return CCR::Equivalent;
    return Res;
  }

  const ValueInfo *getEqualOrEquiv() const {
    return getValueInfo(makeWeakResult(ComparisonCategoryResult::Equal));
  }
  const ValueInfo *getLess() const {
    return getValueInfo(ComparisonCategoryResult::Less);
  }
  const ValueInfo *getGreater() const {
    return getValueInfo(ComparisonCategoryResult::Greater);
  }
  const ValueInfo *getUnordered() const {
    assert(isPartial());
    return getValueInfo(ComparisonCategoryResult::Unordered);
  }
};

class ComparisonCategories {
public:
  static StringRef getCategoryString(ComparisonCategoryType Kind);
  static StringRef getResultString(ComparisonCategoryResult Kind);

  /// Return the list of results which are valid for the specified
  /// comparison category type.
  static std::vector<ComparisonCategoryResult>
  getPossibleResultsForType(ComparisonCategoryType Type);

  /// Return the comparison category information for the category
  /// specified by 'Kind'.
  const ComparisonCategoryInfo &getInfo(ComparisonCategoryType Kind) const {
    const ComparisonCategoryInfo *Result = lookupInfo(Kind);
    assert(Result != nullptr &&
           "information for specified comparison category has not been built");
    return *Result;
  }

  /// Return the comparison category information as specified by
  /// `getCategoryForType(Ty)`. If the information is not already cached,
  /// the declaration is looked up and a cache entry is created.
  /// NOTE: Lookup is expected to succeed. Use lookupInfo if failure is
  /// possible.
  const ComparisonCategoryInfo &getInfoForType(QualType Ty) const;

public:
  /// Return the cached comparison category information for the
  /// specified 'Kind'. If no cache entry is present the comparison category
  /// type is looked up. If lookup fails nullptr is returned. Otherwise, a
  /// new cache entry is created and returned
  const ComparisonCategoryInfo *lookupInfo(ComparisonCategoryType Kind) const;

  ComparisonCategoryInfo *lookupInfo(ComparisonCategoryType Kind) {
    const auto &This = *this;
    return const_cast<ComparisonCategoryInfo *>(This.lookupInfo(Kind));
  }

  const ComparisonCategoryInfo *lookupInfoForType(QualType Ty) const;

private:
  friend class ASTContext;

  explicit ComparisonCategories(const ASTContext &Ctx) : Ctx(Ctx) {}

  const ASTContext &Ctx;

  /// A map from the ComparisonCategoryType (represented as 'char') to the
  /// cached information for the specified category.
  mutable toolchain::DenseMap<char, ComparisonCategoryInfo> Data;
  mutable NamespaceDecl *StdNS = nullptr;
};

} // namespace language::Core

#endif
