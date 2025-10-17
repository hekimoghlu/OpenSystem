/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The ValidateVaryingLocations function checks if there exists location conflicts on shader
// varyings.
//

#ifndef COMPILER_TRANSLATOR_VALIDATEVARYINGLOCATIONS_H_
#define COMPILER_TRANSLATOR_VALIDATEVARYINGLOCATIONS_H_

#include "GLSLANG/ShaderVars.h"

namespace sh
{

class TIntermBlock;
class TIntermSymbol;
class TDiagnostics;
class TType;

unsigned int CalculateVaryingLocationCount(const TType &varyingType, GLenum shaderType);
bool ValidateVaryingLocations(TIntermBlock *root, TDiagnostics *diagnostics, GLenum shaderType);

}  // namespace sh

#endif
