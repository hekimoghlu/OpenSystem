/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

struct AudioConfiguration {
    String contentType;
    String channels;
    std::optional<uint64_t> bitrate;
    std::optional<uint32_t> samplerate;
    std::optional<bool> spatialRendering;

    AudioConfiguration isolatedCopy() const &;
    AudioConfiguration isolatedCopy() &&;
};

inline AudioConfiguration AudioConfiguration::isolatedCopy() const &
{
    return { contentType.isolatedCopy(), channels.isolatedCopy(), bitrate, samplerate, spatialRendering };
}

inline AudioConfiguration AudioConfiguration::isolatedCopy() &&
{
    return { WTFMove(contentType).isolatedCopy(), WTFMove(channels).isolatedCopy(), bitrate, samplerate, spatialRendering };
}

} // namespace WebCore
