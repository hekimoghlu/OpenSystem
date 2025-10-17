/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

//===--- SILParserFunctionBuilder.h ---------------------------------------===//
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

#ifndef LANGUAGE_PARSESIL_SILPARSERFUNCTIONBUILDER_H
#define LANGUAGE_PARSESIL_SILPARSERFUNCTIONBUILDER_H

#include "language/SIL/SILFunctionBuilder.h"

namespace language {

class TOOLCHAIN_LIBRARY_VISIBILITY SILParserFunctionBuilder {
  SILFunctionBuilder builder;

public:
  SILParserFunctionBuilder(SILModule &mod) : builder(mod) {}

  SILFunction *createFunctionForForwardReference(StringRef name,
                                                 CanSILFunctionType ty,
                                                 SILLocation loc) {
    auto *result = builder.createFunction(
        SILLinkage::Private, name, ty, nullptr, loc, IsNotBare,
        IsNotTransparent, IsNotSerialized, IsNotDynamic, IsNotDistributed,
        IsNotRuntimeAccessible);
    result->setDebugScope(new (builder.mod) SILDebugScope(loc, result));

    // If we did not have a declcontext set, as a fallback set the parent module
    // of our SILFunction to the parent module of our SILModule.
    //
    // DISCUSSION: This ensures that we can perform protocol conformance checks.
    if (!result->getDeclContext()) {
      result->setParentModule(result->getModule().getCodiraModule());
    }

    return result;
  }
};

} // namespace language

#endif
