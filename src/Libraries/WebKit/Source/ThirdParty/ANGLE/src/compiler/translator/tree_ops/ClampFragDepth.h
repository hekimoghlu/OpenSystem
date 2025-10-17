/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ClampFragDepth.h: Limit the value that is written to gl_FragDepth to the range [0.0, 1.0].
// The clamping is run at the very end of shader execution, and is only performed if the shader
// statically accesses gl_FragDepth.
//

#ifndef COMPILER_TRANSLATOR_TREEOPS_CLAMPFRAGDEPTH_H_
#define COMPILER_TRANSLATOR_TREEOPS_CLAMPFRAGDEPTH_H_

#include "common/angleutils.h"
#include "common/debug.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TSymbolTable;

[[nodiscard]] bool ClampFragDepth(TCompiler *compiler,
                                  TIntermBlock *root,
                                  TSymbolTable *symbolTable);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_CLAMPFRAGDEPTH_H_
