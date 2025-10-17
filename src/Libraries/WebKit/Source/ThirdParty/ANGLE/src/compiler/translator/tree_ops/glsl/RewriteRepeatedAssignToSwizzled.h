/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteRepeatedAssignToSwizzled.h: Rewrite expressions that assign an assignment to a swizzled
// vector, like:
//     v.x = z = expression;
// to:
//     z = expression;
//     v.x = z;
//
// Note that this doesn't handle some corner cases: expressions nested inside other expressions,
// inside loop headers, or inside if conditions.

#ifndef COMPILER_TRANSLATOR_TREEOPS_GLSL_REWRITEREPEATEDASSIGNTOSWIZZLED_H_
#define COMPILER_TRANSLATOR_TREEOPS_GLSL_REWRITEREPEATEDASSIGNTOSWIZZLED_H_

#include "common/angleutils.h"
#include "common/debug.h"

namespace sh
{

class TCompiler;
class TIntermBlock;

#ifdef ANGLE_ENABLE_GLSL
[[nodiscard]] bool RewriteRepeatedAssignToSwizzled(TCompiler *compiler, TIntermBlock *root);
#else
[[nodiscard]] ANGLE_INLINE bool RewriteRepeatedAssignToSwizzled(TCompiler *compiler,
                                                                TIntermBlock *root)
{
    UNREACHABLE();
    return false;
}
#endif

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_GLSL_REWRITEREPEATEDASSIGNTOSWIZZLED_H_
