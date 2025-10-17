/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

// ShaderExecutable9.cpp: Implements a D3D9-specific class to contain shader
// executable implementation details.

#include "libANGLE/renderer/d3d/d3d9/ShaderExecutable9.h"

#include "common/debug.h"

namespace rx
{

ShaderExecutable9::ShaderExecutable9(const void *function,
                                     size_t length,
                                     IDirect3DPixelShader9 *executable)
    : ShaderExecutableD3D(function, length)
{
    mPixelExecutable  = executable;
    mVertexExecutable = nullptr;
}

ShaderExecutable9::ShaderExecutable9(const void *function,
                                     size_t length,
                                     IDirect3DVertexShader9 *executable)
    : ShaderExecutableD3D(function, length)
{
    mVertexExecutable = executable;
    mPixelExecutable  = nullptr;
}

ShaderExecutable9::~ShaderExecutable9()
{
    SafeRelease(mVertexExecutable);
    SafeRelease(mPixelExecutable);
}

IDirect3DVertexShader9 *ShaderExecutable9::getVertexShader() const
{
    return mVertexExecutable;
}

IDirect3DPixelShader9 *ShaderExecutable9::getPixelShader() const
{
    return mPixelExecutable;
}

}  // namespace rx
