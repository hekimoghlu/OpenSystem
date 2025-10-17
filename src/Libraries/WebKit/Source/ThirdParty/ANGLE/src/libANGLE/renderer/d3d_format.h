/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
// d3d_format: Describes a D3D9 format. Used by the D3D9 and GL back-ends.

#ifndef LIBANGLE_RENDERER_D3D_FORMAT_H_
#define LIBANGLE_RENDERER_D3D_FORMAT_H_

#include "libANGLE/renderer/Format.h"

// We forcibly include the D3D9 header here because this file can be used from 3 different backends.
#include <d3d9.h>

namespace rx
{
namespace d3d9
{
struct D3DFormat
{
    D3DFormat();
    D3DFormat(GLuint pixelBytes,
              GLuint blockWidth,
              GLuint blockHeight,
              GLuint redBits,
              GLuint greenBits,
              GLuint blueBits,
              GLuint alphaBits,
              GLuint luminanceBits,
              GLuint depthBits,
              GLuint stencilBits,
              angle::FormatID formatID);

    const angle::Format &info() const { return angle::Format::Get(formatID); }

    GLuint pixelBytes;
    GLuint blockWidth;
    GLuint blockHeight;

    GLuint redBits;
    GLuint greenBits;
    GLuint blueBits;
    GLuint alphaBits;
    GLuint luminanceBits;

    GLuint depthBits;
    GLuint stencilBits;

    angle::FormatID formatID;
};

const D3DFormat &GetD3DFormatInfo(D3DFORMAT format);

}  // namespace d3d9
}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_FORMAT_H_
