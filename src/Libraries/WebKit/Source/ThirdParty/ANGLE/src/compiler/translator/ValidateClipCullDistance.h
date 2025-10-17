/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The ValidateClipCullDistance function checks if the sum of array sizes for gl_ClipDistance and
// gl_CullDistance exceeds gl_MaxCombinedClipAndCullDistances
//

#ifndef COMPILER_TRANSLATOR_VALIDATECLIPCULLDISTANCE_H_
#define COMPILER_TRANSLATOR_VALIDATECLIPCULLDISTANCE_H_

#include "GLSLANG/ShaderVars.h"
#include "compiler/translator/Compiler.h"

namespace sh
{

class TIntermBlock;
class TDiagnostics;

bool ValidateClipCullDistance(TCompiler *compiler,
                              TIntermBlock *root,
                              TDiagnostics *diagnostics,
                              const unsigned int maxCombinedClipAndCullDistances,
                              uint8_t *clipDistanceSizeOut,
                              uint8_t *cullDistanceSizeOut,
                              bool *clipDistanceUsedOut);

}  // namespace sh

#endif
