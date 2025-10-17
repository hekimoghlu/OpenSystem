/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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
// RunAtTheBeginningOfShader.h: Add code to be run at the beginning of the shader.
//

#ifndef COMPILER_TRANSLATOR_TREEUTIL_RUNATTHEBEGINNINGOFSHADER_H_
#define COMPILER_TRANSLATOR_TREEUTIL_RUNATTHEBEGINNINGOFSHADER_H_

#include "common/angleutils.h"

namespace sh
{

class TCompiler;
class TIntermBlock;
class TIntermNode;

[[nodiscard]] bool RunAtTheBeginningOfShader(TCompiler *compiler,
                                             TIntermBlock *root,
                                             TIntermNode *codeToRun);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_RUNATTHEBEGINNINGOFSHADER_H_
