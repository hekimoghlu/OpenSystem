/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// wgpu_wgsl_util.h: Utilities to manipulate previously translated WGSL.
//

#ifndef LIBANGLE_RENDERER_WGPU_WGPU_WGSL_UTIL_H_
#define LIBANGLE_RENDERER_WGPU_WGPU_WGSL_UTIL_H_

#include "common/PackedGLEnums_autogen.h"
#include "libANGLE/Program.h"

namespace rx
{
namespace webgpu
{

// Replaces location markers in the WGSL source with actual locations, for
// `shaderVars` which is a vector of either gl::ProgramInputs or gl::ProgramOutputs, and for
// `mergedVaryings` which get assigned sequentially increasing locations. There should be at most
// vertex and fragment shader stages or this function will not assign locations correctly.
template <typename T>
std::string WgslAssignLocations(const std::string &shaderSource,
                                const std::vector<T> shaderVars,
                                const gl::ProgramMergedVaryings &mergedVaryings,
                                gl::ShaderType shaderType);

}  // namespace webgpu
}  // namespace rx

#endif /* LIBANGLE_RENDERER_WGPU_WGPU_WGSL_UTIL_H_ */
