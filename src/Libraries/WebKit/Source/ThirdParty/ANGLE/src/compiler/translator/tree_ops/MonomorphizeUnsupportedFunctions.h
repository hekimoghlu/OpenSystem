/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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

//
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// MonomorphizeUnsupportedFunctions: Monomorphize functions that are called with
// parameters that are incompatible with both Vulkan GLSL and Metal:
//
// - Samplers in structs
// - Structs that have samplers
// - Partially subscripted array of array of samplers
// - Partially subscripted array of array of images
// - Atomic counters
// - samplerCube variables when emulating ES2's cube map sampling
// - image* variables with r32f formats (to emulate imageAtomicExchange)
//
// This transformation basically duplicates such functions, removes the
// sampler/image/atomic_counter parameters and uses the opaque uniforms used by the caller.

#ifndef COMPILER_TRANSLATOR_TREEOPS_MONOMORPHIZEUNSUPPORTEDFUNCTIONS_H_
#define COMPILER_TRANSLATOR_TREEOPS_MONOMORPHIZEUNSUPPORTEDFUNCTIONS_H_

#include "common/angleutils.h"
#include "compiler/translator/Compiler.h"

namespace sh
{
class TIntermBlock;
class TSymbolTable;

// Types of function prameters that should trigger monomorphization.
enum class UnsupportedFunctionArgs
{
    StructContainingSamplers     = 0,
    ArrayOfArrayOfSamplerOrImage = 1,
    AtomicCounter                = 2,
    Image                        = 3,
    PixelLocalStorage            = 4,

    InvalidEnum = 6,
    EnumCount   = 6,
};

using UnsupportedFunctionArgsBitSet = angle::PackedEnumBitSet<UnsupportedFunctionArgs>;

[[nodiscard]] bool MonomorphizeUnsupportedFunctions(TCompiler *compiler,
                                                    TIntermBlock *root,
                                                    TSymbolTable *symbolTable,
                                                    UnsupportedFunctionArgsBitSet);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_MONOMORPHIZEUNSUPPORTEDFUNCTIONS_H_
