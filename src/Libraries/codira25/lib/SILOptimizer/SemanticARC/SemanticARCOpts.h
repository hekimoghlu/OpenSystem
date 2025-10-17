/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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

//===--- SemanticARCOpts.h ------------------------------------------------===//
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

#ifndef LANGUAGE_SILOPTIMIZER_SEMANTICARC_SEMANTICARCOPTS_H
#define LANGUAGE_SILOPTIMIZER_SEMANTICARC_SEMANTICARCOPTS_H

#include <cstdint>
#include <type_traits>

namespace language {
namespace semanticarc {

/// An enum used so that at the command line, we can override which transforms
/// we perform.
enum class ARCTransformKind : uint64_t {
  Invalid = 0,
  OwnedToGuaranteedPhi = 0x1,
  LoadCopyToLoadBorrowPeephole = 0x2,
  RedundantBorrowScopeElimPeephole = 0x4,
  // TODO: Split RedundantCopyValueElimPeephole into more granular categories
  // such as dead live range, guaranteed copy_value opt, etc.
  RedundantCopyValueElimPeephole = 0x8,
  LifetimeJoiningPeephole = 0x10,
  OwnershipConversionElimPeephole = 0x20,
  RedundantMoveValueElim = 0x40,

  AllPeepholes = LoadCopyToLoadBorrowPeephole |
                 RedundantBorrowScopeElimPeephole |
                 RedundantCopyValueElimPeephole | LifetimeJoiningPeephole |
                 OwnershipConversionElimPeephole,
  All = AllPeepholes | OwnedToGuaranteedPhi | RedundantMoveValueElim,
};

inline ARCTransformKind operator&(ARCTransformKind lhs, ARCTransformKind rhs) {
  using UnderlyingTy = std::underlying_type<ARCTransformKind>::type;
  return ARCTransformKind(UnderlyingTy(lhs) & UnderlyingTy(rhs));
}

} // namespace semanticarc
} // namespace language

#endif
