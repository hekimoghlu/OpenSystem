/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

//===--- GenBuiltin.h - IR generation for builtin functions -----*- C++ -*-===//
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
//
//  This file provides the private interface to the emission of builtin
//  functions in Codira.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENBUILTIN_H
#define LANGUAGE_IRGEN_GENBUILTIN_H

#include "language/AST/SubstitutionMap.h"
#include "language/Basic/Toolchain.h"

namespace language {
  class BuiltinInfo;
  class BuiltinInst;
  class Identifier;
  class SILType;

namespace irgen {
  class Explosion;
  class IRGenFunction;

  /// Emit a call to a builtin function.
  void emitBuiltinCall(IRGenFunction &IGF, const BuiltinInfo &builtin,
                       BuiltinInst *Inst, ArrayRef<SILType> argTypes,
                       Explosion &args, Explosion &result);

} // end namespace irgen
} // end namespace language

#endif
