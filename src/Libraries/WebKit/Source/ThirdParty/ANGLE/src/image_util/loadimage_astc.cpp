/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 9, 2025.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// loadimage_astc.cpp: Decodes ASTC encoded textures.

#include "image_util/AstcDecompressor.h"
#include "image_util/loadimage.h"

namespace angle
{

void LoadASTCToRGBA8Inner(const ImageLoadContext &context,
                          size_t width,
                          size_t height,
                          size_t depth,
                          uint32_t blockWidth,
                          uint32_t blockHeight,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    auto imgWidth  = static_cast<uint32_t>(width);
    auto imgHeight = static_cast<uint32_t>(height);

    AstcDecompressor &decompressor = AstcDecompressor::get();
    if (!decompressor.available())
    {
        ERR() << "Trying to decompress ASTC without having ASTC support built.";
        return;
    }

    // Compute the number of ASTC blocks in each dimension
    uint32_t blockCountX = (imgWidth + blockWidth - 1) / blockWidth;
    uint32_t blockCountY = (imgHeight + blockHeight - 1) / blockHeight;

    // Space needed for 16 bytes of output per compressed block
    size_t blockSize = blockCountX * blockCountY * 16;

    int32_t result =
        decompressor.decompress(context.singleThreadPool, context.multiThreadPool, imgWidth,
                                imgHeight, blockWidth, blockHeight, input, blockSize, output);
    if (result != 0)
    {
        WARN() << "ASTC decompression failed: " << decompressor.getStatusString(result);
    }
}
}  // namespace angle
