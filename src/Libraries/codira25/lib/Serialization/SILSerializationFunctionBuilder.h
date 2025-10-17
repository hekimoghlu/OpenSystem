/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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

//===--- SILSerializationFunctionBuilder.h --------------------------------===//
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

#ifndef LANGUAGE_SERIALIZATION_SERIALIZATIONFUNCTIONBUILDER_H
#define LANGUAGE_SERIALIZATION_SERIALIZATIONFUNCTIONBUILDER_H

#include "language/SIL/SILFunctionBuilder.h"

namespace language {

class TOOLCHAIN_LIBRARY_VISIBILITY SILSerializationFunctionBuilder {
  SILFunctionBuilder builder;

public:
  SILSerializationFunctionBuilder(SILModule &mod) : builder(mod) {}

  /// Create a SILFunction declaration for use either as a forward reference or
  /// for the eventual deserialization of a function body.
  SILFunction *createDeclaration(StringRef name, SILType type,
                                 SILLocation loc) {
    return builder.createFunction(
        SILLinkage::Private, name, type.getAs<SILFunctionType>(), nullptr,
        loc, IsNotBare, IsNotTransparent,
        IsNotSerialized, IsNotDynamic, IsNotDistributed, IsNotRuntimeAccessible,
        ProfileCounter(), IsNotThunk, SubclassScope::NotApplicable);
  }

  void setHasOwnership(SILFunction *f, bool newValue) {
    builder.setHasOwnership(f, newValue);
  }
};

} // namespace language

#endif
