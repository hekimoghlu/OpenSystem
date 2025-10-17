/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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

//===--- TypeDifference.h - Utility for concrete type unification ---------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "Symbol.h"
#include "Term.h"
#include "language/AST/Type.h"
#include "toolchain/ADT/ArrayRef.h"
#include "toolchain/ADT/SmallVector.h"
#include <optional>

#ifndef TYPE_DIFFERENCE_H_
#define TYPE_DIFFERENCE_H_

namespace toolchain {

class raw_ostream;

} // end namespace toolchain

namespace language {

namespace rewriting {

class RewriteContext;

/// Describes transformations that turn LHS into RHS, given that there are a
/// pair of rules (BaseTerm.[LHS] => BaseTerm) and (BaseTerm.[RHS] => BaseTerm).
///
/// There are two kinds of transformations:
///
/// - Replacing a type term T1 with another type term T2, where T2 < T1.
/// - Replacing a type term T1 with a concrete type C2.
struct TypeDifference {
  Term BaseTerm;
  Symbol LHS;
  Symbol RHS;

  /// A pair (N1, N2) where N1 is an index into LHS.getSubstitutions() and
  /// N2 is an index into RHS.getSubstitutions().
  SmallVector<std::pair<unsigned, Term>, 1> SameTypes;

  /// A pair (N1, C2) where N1 is an index into LHS.getSubstitutions() and
  /// C2 is a concrete type symbol.
  SmallVector<std::pair<unsigned, Symbol>, 1> ConcreteTypes;

  TypeDifference(Term baseTerm, Symbol lhs, Symbol rhs,
                 SmallVector<std::pair<unsigned, Term>, 1> sameTypes,
                 SmallVector<std::pair<unsigned, Symbol>, 1> concreteTypes)
    : BaseTerm(baseTerm), LHS(lhs), RHS(rhs),
      SameTypes(sameTypes), ConcreteTypes(concreteTypes) {}

  MutableTerm getOriginalSubstitution(unsigned index) const;
  MutableTerm getReplacementSubstitution(unsigned index) const;

  void dump(toolchain::raw_ostream &out) const;
  void verify(RewriteContext &ctx) const;
};

TypeDifference
buildTypeDifference(
    Term baseTerm, Symbol symbol,
    const toolchain::SmallVector<std::pair<unsigned, Term>, 1> &sameTypes,
    const toolchain::SmallVector<std::pair<unsigned, Symbol>, 1> &concreteTypes,
    RewriteContext &ctx);

} // end namespace rewriting

} // end namespace language

#endif