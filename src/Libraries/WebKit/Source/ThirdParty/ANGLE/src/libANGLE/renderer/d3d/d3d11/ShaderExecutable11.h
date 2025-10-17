/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShaderExecutable11.h: Defines a D3D11-specific class to contain shader
// executable implementation details.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_SHADEREXECUTABLE11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_SHADEREXECUTABLE11_H_

#include "libANGLE/renderer/d3d/ShaderExecutableD3D.h"
#include "libANGLE/renderer/d3d/d3d11/ResourceManager11.h"

namespace gl
{
class Context;
}  // namespace gl

namespace rx
{
class Renderer11;
class UniformStorage11;

class ShaderExecutable11 : public ShaderExecutableD3D
{
  public:
    ShaderExecutable11(const void *function, size_t length, d3d11::PixelShader &&executable);
    ShaderExecutable11(const void *function,
                       size_t length,
                       d3d11::VertexShader &&executable,
                       d3d11::GeometryShader &&streamOut);
    ShaderExecutable11(const void *function, size_t length, d3d11::GeometryShader &&executable);
    ShaderExecutable11(const void *function, size_t length, d3d11::ComputeShader &&executable);

    ~ShaderExecutable11() override;

    const d3d11::PixelShader &getPixelShader() const;
    const d3d11::VertexShader &getVertexShader() const;
    const d3d11::GeometryShader &getGeometryShader() const;
    const d3d11::GeometryShader &getStreamOutShader() const;
    const d3d11::ComputeShader &getComputeShader() const;

  private:
    d3d11::PixelShader mPixelExecutable;
    d3d11::VertexShader mVertexExecutable;
    d3d11::GeometryShader mGeometryExecutable;
    d3d11::GeometryShader mStreamOutExecutable;
    d3d11::ComputeShader mComputeExecutable;
};

class UniformStorage11 : public UniformStorageD3D
{
  public:
    UniformStorage11(size_t initialSize);
    ~UniformStorage11() override;

    angle::Result getConstantBuffer(const gl::Context *context,
                                    Renderer11 *renderer,
                                    const d3d11::Buffer **bufferOut);

  private:
    d3d11::Buffer mConstantBuffer;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_SHADEREXECUTABLE11_H_
