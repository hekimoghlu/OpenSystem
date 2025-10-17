/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

//===--- Callee.h -----------------------------------------------*- C++ -*-===//
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

#ifndef LANGUAGE_SILGEN_CALLEE_H
#define LANGUAGE_SILGEN_CALLEE_H

#include "language/AST/ForeignAsyncConvention.h"
#include "language/AST/ForeignErrorConvention.h"
#include "language/AST/ForeignInfo.h"
#include "language/AST/Types.h"
#include "language/SIL/AbstractionPattern.h"

namespace language {
namespace Lowering {

class CalleeTypeInfo {
public:
  std::optional<AbstractionPattern> origFormalType;
  CanSILFunctionType substFnType;
  std::optional<AbstractionPattern> origResultType;
  CanType substResultType;
  ForeignInfo foreign;

private:
  std::optional<SILFunctionTypeRepresentation> overrideRep;

public:
  CalleeTypeInfo() = default;

  CalleeTypeInfo(
      CanSILFunctionType substFnType, AbstractionPattern origResultType,
      CanType substResultType,
      const std::optional<ForeignErrorConvention> &foreignError,
      const std::optional<ForeignAsyncConvention> &foreignAsync,
      ImportAsMemberStatus foreignSelf,
      std::optional<SILFunctionTypeRepresentation> overrideRep = std::nullopt)
      : origFormalType(std::nullopt), substFnType(substFnType),
        origResultType(origResultType), substResultType(substResultType),
        foreign{foreignSelf, foreignError, foreignAsync},
        overrideRep(overrideRep) {}

  CalleeTypeInfo(
      CanSILFunctionType substFnType, AbstractionPattern origResultType,
      CanType substResultType,
      std::optional<SILFunctionTypeRepresentation> overrideRep = std::nullopt)
      : origFormalType(std::nullopt), substFnType(substFnType),
        origResultType(origResultType), substResultType(substResultType),
        foreign(), overrideRep(overrideRep) {}

  SILFunctionTypeRepresentation getOverrideRep() const {
    return overrideRep.value_or(substFnType->getRepresentation());
  }
};

} // namespace Lowering
} // namespace language

#endif
