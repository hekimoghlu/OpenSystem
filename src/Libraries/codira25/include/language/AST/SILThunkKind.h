/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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

//===--- SILThunkKind.h ---------------------------------------------------===//
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
///
/// \file This file contains a kind that defines the types of thunks that can be
/// generated using a thunk inst. It provides an AST level interface that lets
/// one generate the derived function kind and is also used to make adding such
/// kinds to the mangler trivial.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_SILTHUNKKIND_H
#define LANGUAGE_AST_SILTHUNKKIND_H

#include "language/AST/Types.h"

namespace language {

class SILType;

struct SILThunkKind {
  enum InnerTy {
    Invalid = 0,

    /// A thunk that just calls the passed in function. Used for testing
    /// purposes of the underlying thunking machinery.
    Identity = 1,

    MaxValue = Identity,
  };

  InnerTy innerTy;

  SILThunkKind() : innerTy(InnerTy::Invalid) {}
  SILThunkKind(InnerTy innerTy) : innerTy(innerTy) {}
  SILThunkKind(unsigned inputInnerTy) : innerTy(InnerTy(inputInnerTy)) {
    assert(inputInnerTy <= MaxValue && "Invalid value");
  }

  operator InnerTy() const { return innerTy; }

  /// Given the current enum state returned the derived output function from
  /// \p inputFunction.
  ///
  /// Defined in Instructions.cpp
  CanSILFunctionType getDerivedFunctionType(SILFunction *fn,
                                            CanSILFunctionType inputFunction,
                                            SubstitutionMap subMap) const;

  /// Given the current enum state returned the derived output function from
  /// \p inputFunction.
  ///
  /// Defined in Instructions.cpp
  SILType getDerivedFunctionType(SILFunction *fn, SILType inputFunctionType,
                                 SubstitutionMap subMap) const;

  Demangle::MangledSILThunkKind getMangledKind() const {
    switch (innerTy) {
    case Invalid:
      return Demangle::MangledSILThunkKind::Invalid;
    case Identity:
      return Demangle::MangledSILThunkKind::Identity;
    }
  }
};

} // namespace language

#endif
