/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DynamicImage2DHLSL.h: Interface for link and run-time HLSL generation
//

#ifndef LIBANGLE_RENDERER_D3D_DYNAMICIMAGE2DHLSL_H_
#define LIBANGLE_RENDERER_D3D_DYNAMICIMAGE2DHLSL_H_

#include "common/angleutils.h"
#include "libANGLE/renderer/d3d/RendererD3D.h"
#include "libANGLE/renderer/d3d/ShaderD3D.h"

namespace rx
{
std::string GenerateShaderForImage2DBindSignatureImpl(
    ProgramExecutableD3D &executableD3D,
    gl::ShaderType shaderType,
    const SharedCompiledShaderStateD3D &shaderData,
    const std::string &shaderHLSL,
    std::vector<sh::ShaderVariable> &image2DUniforms,
    const gl::ImageUnitTextureTypeMap &image2DBindLayout,
    unsigned int baseUAVRegister);

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_DYNAMICHLSL_H_
