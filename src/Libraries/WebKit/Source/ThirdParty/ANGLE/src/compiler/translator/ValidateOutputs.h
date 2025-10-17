/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ValidateOutputs validates fragment shader outputs. It checks for conflicting locations,
// out-of-range locations, that locations are specified when using multiple outputs, and YUV output
// validity.
//

#ifndef COMPILER_TRANSLATOR_VALIDATEOUTPUTS_H_
#define COMPILER_TRANSLATOR_VALIDATEOUTPUTS_H_

#include <GLSLANG/ShaderLang.h>

#include "compiler/translator/ExtensionBehavior.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TDiagnostics;

// Returns true if the shader has no conflicting or otherwise erroneous fragment outputs.
bool ValidateOutputs(TIntermBlock *root,
                     const TExtensionBehavior &extBehavior,
                     const ShBuiltInResources &resources,
                     bool usesPixelLocalStorage,
                     bool isWebGL,
                     TDiagnostics *diagnostics);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VALIDATEOUTPUTS_H_
