/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#pragma once

#include "ColorGamut.h"
#include "HdrMetadataType.h"
#include "TransferFunction.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

struct VideoConfiguration {
    String contentType;
    uint32_t width;
    uint32_t height;
    uint64_t bitrate;
    double framerate;
    std::optional<bool> alphaChannel;
    std::optional<ColorGamut> colorGamut;
    std::optional<HdrMetadataType> hdrMetadataType;
    std::optional<TransferFunction> transferFunction;

    VideoConfiguration isolatedCopy() const &;
    VideoConfiguration isolatedCopy() &&;
};

inline VideoConfiguration VideoConfiguration::isolatedCopy() const &
{
    return { contentType.isolatedCopy(), width, height, bitrate, framerate, alphaChannel, colorGamut, hdrMetadataType, transferFunction };
}

inline VideoConfiguration VideoConfiguration::isolatedCopy() &&
{
    return { WTFMove(contentType).isolatedCopy(), width, height, bitrate, framerate, alphaChannel, colorGamut, hdrMetadataType, transferFunction };
}

} // namespace WebCore
