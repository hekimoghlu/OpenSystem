/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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

//===--- GenKeyPath.h - IR generation for KeyPath ---------------*- C++ -*-===//
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
//  This file provides the private interface to the emission of KeyPath
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENKEYPATH_H
#define LANGUAGE_IRGEN_GENKEYPATH_H

#include "GenericRequirement.h"
#include "language/AST/SubstitutionMap.h"
#include "language/Basic/Toolchain.h"
#include "language/SIL/SILValue.h"
#include "toolchain/IR/Value.h"

namespace language {
namespace irgen {
class Explosion;
class IRGenFunction;
class StackAddress;

std::pair<toolchain::Value *, toolchain::Value *>
emitKeyPathArgument(IRGenFunction &IGF, SubstitutionMap subs,
                    const CanGenericSignature &sig,
                    ArrayRef<SILType> indiceTypes, Explosion &indiceValues,
                    std::optional<StackAddress> &dynamicArgsBuf);
} // end namespace irgen
} // end namespace language

#endif
