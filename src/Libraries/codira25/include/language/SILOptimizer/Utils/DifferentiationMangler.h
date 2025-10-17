/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

//===------- DifferentiationMangler.h --------- differentiation -*- C++ -*-===//
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

#ifndef LANGUAGE_SIL_UTILS_DIFFERENTIATIONMANGLER_H
#define LANGUAGE_SIL_UTILS_DIFFERENTIATIONMANGLER_H

#include "language/AST/ASTContext.h"
#include "language/AST/ASTMangler.h"
#include "language/AST/AutoDiff.h"
#include "language/Basic/NullablePtr.h"
#include "language/Demangling/Demangler.h"
#include "language/SIL/SILFunction.h"

namespace language {
namespace Mangle {

/// A mangler for generated differentiation functions.
class DifferentiationMangler : public ASTMangler {
public:
  DifferentiationMangler(const ASTContext &Ctx) : ASTMangler(Ctx) {}
  /// Returns the mangled name for a differentiation function of the given kind.
  std::string mangleAutoDiffFunction(StringRef originalName,
                                     Demangle::AutoDiffFunctionKind kind,
                                     const AutoDiffConfig &config);
  /// Returns the mangled name for a derivative function of the given kind.
  std::string mangleDerivativeFunction(StringRef originalName,
                                       AutoDiffDerivativeFunctionKind kind,
                                       const AutoDiffConfig &config);
  /// Returns the mangled name for a linear map of the given kind.
  std::string mangleLinearMap(StringRef originalName,
                              AutoDiffLinearMapKind kind,
                              const AutoDiffConfig &config);
  /// Returns the mangled name for a derivative function subset parameters
  /// thunk.
  std::string mangleDerivativeFunctionSubsetParametersThunk(
      StringRef originalName, CanType toType,
      AutoDiffDerivativeFunctionKind linearMapKind,
      IndexSubset *fromParamIndices, IndexSubset *fromResultIndices,
      IndexSubset *toParamIndices);
  /// Returns the mangled name for a linear map subset parameters thunk.
  std::string mangleLinearMapSubsetParametersThunk(
      CanType fromType, AutoDiffLinearMapKind linearMapKind,
      IndexSubset *fromParamIndices, IndexSubset *fromResultIndices,
      IndexSubset *toParamIndices);
};

} // end namespace Mangle
} // end namespace language

#endif /* LANGUAGE_SIL_UTILS_DIFFERENTIATIONMANGLER_H */
