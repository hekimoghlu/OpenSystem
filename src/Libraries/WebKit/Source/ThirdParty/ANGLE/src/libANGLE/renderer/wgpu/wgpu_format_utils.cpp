/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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

#include "libANGLE/renderer/wgpu/wgpu_format_utils.h"

#include "libANGLE/renderer/load_functions_table.h"
namespace rx
{
namespace
{
void FillTextureCaps(const angle::Format &angleFormat,
                     angle::FormatID formatID,
                     gl::TextureCaps *outTextureCaps)
{
    if (formatID != angle::FormatID::NONE)
    {
        outTextureCaps->texturable = true;
    }
    outTextureCaps->filterable   = true;
    outTextureCaps->renderbuffer = true;
    outTextureCaps->blendable    = true;
}
}  // namespace

namespace webgpu
{
Format::Format()
    : mIntendedFormatID(angle::FormatID::NONE),
      mIntendedGLFormat(GL_NONE),
      mActualImageFormatID(angle::FormatID::NONE),
      mActualBufferFormatID(angle::FormatID::NONE),
      mImageInitializerFunction(nullptr),
      mIsRenderable(false)
{}
void Format::initImageFallback(const ImageFormatInitInfo *info, int numInfo)
{
    UNIMPLEMENTED();
}

void Format::initBufferFallback(const BufferFormatInitInfo *fallbackInfo, int numInfo)
{
    UNIMPLEMENTED();
}

FormatTable::FormatTable() {}
FormatTable::~FormatTable() {}

void FormatTable::initialize()
{
    for (size_t formatIndex = 0; formatIndex < angle::kNumANGLEFormats; ++formatIndex)
    {
        Format &format                           = mFormatData[formatIndex];
        const auto intendedFormatID              = static_cast<angle::FormatID>(formatIndex);
        const angle::Format &intendedAngleFormat = angle::Format::Get(intendedFormatID);

        format.initialize(intendedAngleFormat);
        format.mIntendedFormatID = intendedFormatID;

        gl::TextureCaps textureCaps;
        FillTextureCaps(format.getActualImageFormat(), format.mActualImageFormatID, &textureCaps);
        if (textureCaps.texturable)
        {
            format.mTextureLoadFunctions =
                GetLoadFunctionsMap(format.mIntendedGLFormat, format.mActualImageFormatID);
        }
    }
}
}  // namespace webgpu
}  // namespace rx
