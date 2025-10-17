/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
// Copyright 2023 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// renderermtl_utils:
//   Helper methods pertaining to the Metal backend.
//

#ifndef LIBANGLE_RENDERER_METAL_RENDERERMTL_UTILS_H_
#define LIBANGLE_RENDERER_METAL_RENDERERMTL_UTILS_H_

#include <cstdint>

#include <limits>
#include <map>

#include "GLSLANG/ShaderLang.h"
#include "common/angleutils.h"
#include "common/utilities.h"
#include "libANGLE/angletypes.h"

namespace rx
{
namespace mtl
{

template <int cols, int rows>
struct SetFloatUniformMatrixMetal
{
    static void Run(unsigned int arrayElementOffset,
                    unsigned int elementCount,
                    GLsizei countIn,
                    GLboolean transpose,
                    const GLfloat *value,
                    uint8_t *targetData);
};

// Helper method to de-tranpose a matrix uniform for an API query, Metal Version
void GetMatrixUniformMetal(GLenum type, GLfloat *dataOut, const GLfloat *source, bool transpose);

template <typename NonFloatT>
void GetMatrixUniformMetal(GLenum type,
                           NonFloatT *dataOut,
                           const NonFloatT *source,
                           bool transpose);
}  // namespace mtl
}  // namespace rx
#endif
