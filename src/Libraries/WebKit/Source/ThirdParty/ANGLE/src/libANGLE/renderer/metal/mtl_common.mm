/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
// mtl_common.mm:
//      Implementation of mtl::Context, the MTLDevice container & error handler class.
//

#include "libANGLE/renderer/metal/mtl_common.h"

#include <dispatch/dispatch.h>

#include <cstring>

#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/metal/DisplayMtl.h"

namespace rx
{
namespace mtl
{

// ClearColorValue implementation
ClearColorValue &ClearColorValue::operator=(const ClearColorValue &src)
{
    mType       = src.mType;
    mValueBytes = src.mValueBytes;

    return *this;
}

void ClearColorValue::setAsFloat(float r, float g, float b, float a)
{
    mType   = PixelType::Float;
    mRedF   = r;
    mGreenF = g;
    mBlueF  = b;
    mAlphaF = a;
}

void ClearColorValue::setAsInt(int32_t r, int32_t g, int32_t b, int32_t a)
{
    mType   = PixelType::Int;
    mRedI   = r;
    mGreenI = g;
    mBlueI  = b;
    mAlphaI = a;
}

void ClearColorValue::setAsUInt(uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
    mType   = PixelType::UInt;
    mRedU   = r;
    mGreenU = g;
    mBlueU  = b;
    mAlphaU = a;
}

MTLClearColor ClearColorValue::toMTLClearColor() const
{
    switch (mType)
    {
        case PixelType::Int:
            return MTLClearColorMake(mRedI, mGreenI, mBlueI, mAlphaI);
        case PixelType::UInt:
            return MTLClearColorMake(mRedU, mGreenU, mBlueU, mAlphaU);
        case PixelType::Float:
            return MTLClearColorMake(mRedF, mGreenF, mBlueF, mAlphaF);
        default:
            UNREACHABLE();
            return MTLClearColorMake(0, 0, 0, 0);
    }
}

// ImageNativeIndex implementation
ImageNativeIndexIterator ImageNativeIndex::getLayerIterator(GLint layerCount) const
{
    return ImageNativeIndexIterator(mNativeIndex.getLayerIterator(layerCount));
}

// Context implementation
Context::Context(DisplayMtl *display) : mDisplay(display) {}

mtl::CommandQueue &Context::cmdQueue()
{
    return mDisplay->cmdQueue();
}

}  // namespace mtl
}  // namespace rx
