/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

//===--- SILGenFunctionBuilder.h ------------------------------------------===//
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

#ifndef LANGUAGE_SILGEN_SILGENFUNCTIONBUILDER_H
#define LANGUAGE_SILGEN_SILGENFUNCTIONBUILDER_H

#include "language/SIL/SILFunctionBuilder.h"

namespace language {
namespace Lowering {

class TOOLCHAIN_LIBRARY_VISIBILITY SILGenFunctionBuilder {
  SILFunctionBuilder builder;

public:
  SILGenFunctionBuilder(SILGenModule &SGM) : builder(SGM.M) {}
  SILGenFunctionBuilder(SILGenFunction &SGF) : builder(SGF.SGM.M) {}

  template <class... ArgTys>
  SILFunction *getOrCreateSharedFunction(ArgTys &&... args) {
    return builder.getOrCreateSharedFunction(std::forward<ArgTys>(args)...);
  }

  template <class... ArgTys>
  SILFunction *getOrCreateFunction(ArgTys &&... args) {
    return builder.getOrCreateFunction(std::forward<ArgTys>(args)...);
  }

  template <class... ArgTys> SILFunction *createFunction(ArgTys &&... args) {
    return builder.createFunction(std::forward<ArgTys>(args)...);
  }
};

} // namespace Lowering
} // namespace language

#endif
