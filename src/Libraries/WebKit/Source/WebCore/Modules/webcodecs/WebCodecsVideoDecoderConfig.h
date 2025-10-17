/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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

#if ENABLE(WEB_CODECS)

#include "BufferSource.h"
#include "HardwareAcceleration.h"
#include "VideoColorSpaceInit.h"
#include <optional>

namespace WebCore {

struct WebCodecsVideoDecoderConfig {
    String codec;
    std::optional<BufferSource::VariantType> description;
    std::optional<size_t> codedWidth;
    std::optional<size_t> codedHeight;
    std::optional<size_t> displayAspectWidth;
    std::optional<size_t> displayAspectHeight;
    std::optional<VideoColorSpaceInit> colorSpace;
    HardwareAcceleration hardwareAcceleration { HardwareAcceleration::NoPreference };
    std::optional<bool> optimizeForLatency;

    WebCodecsVideoDecoderConfig isolatedCopyWithoutDescription() && { return { WTFMove(codec).isolatedCopy(), { }, codedWidth, codedHeight, displayAspectWidth, displayAspectHeight, colorSpace, hardwareAcceleration, optimizeForLatency }; }
    WebCodecsVideoDecoderConfig isolatedCopyWithoutDescription() const & { return { codec.isolatedCopy(), { }, codedWidth, codedHeight, displayAspectWidth, displayAspectHeight, colorSpace, hardwareAcceleration, optimizeForLatency }; }
};

}

#endif
