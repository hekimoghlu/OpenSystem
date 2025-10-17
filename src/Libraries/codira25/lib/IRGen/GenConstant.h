/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 4, 2024.
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

//===--- GenConstant.h - Codira IR Generation For Constants ------*- C++ -*-===//
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
//  This file implements IR generation for constant values.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_IRGEN_GENCONSTANT_H
#define LANGUAGE_IRGEN_GENCONSTANT_H

#include "toolchain/IR/Constant.h"

#include "IRGenModule.h"

namespace language {
namespace irgen {

/// Construct a ConstantInt from an IntegerLiteralInst.
toolchain::Constant *emitConstantInt(IRGenModule &IGM, IntegerLiteralInst *ILI);

/// Construct a zero from a zero initializer BuiltinInst.
toolchain::Constant *emitConstantZero(IRGenModule &IGM, BuiltinInst *Bi);

/// Construct a ConstantFP from a FloatLiteralInst.
toolchain::Constant *emitConstantFP(IRGenModule &IGM, FloatLiteralInst *FLI);

/// Construct a pointer to a string from a StringLiteralInst.
toolchain::Constant *emitAddrOfConstantString(IRGenModule &IGM,
                                         StringLiteralInst *SLI);

/// Construct a constant from a SILValue containing constant values.
Explosion emitConstantValue(IRGenModule &IGM, SILValue value,
                            bool flatten = false);

/// Construct an object (with a HeapObject header) from an ObjectInst
/// containing constant values.
toolchain::Constant *emitConstantObject(IRGenModule &IGM, ObjectInst *OI,
                                   StructLayout *ClassLayout);
}
}

#endif
