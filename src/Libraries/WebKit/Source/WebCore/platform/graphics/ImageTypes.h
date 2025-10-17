/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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

#if USE(CG)
typedef struct CGImageSource* CGImageSourceRef;
#endif

namespace WebCore {

// There are four subsampling levels: 0 = 1x, 1 = 0.5x, 2 = 0.25x, 3 = 0.125x.
enum class SubsamplingLevel {
    First = 0,
    Default = First,
    Level0 = First,
    Level1,
    Level2,
    Level3,
    Last = Level3,
    Max
};

inline SubsamplingLevel& operator++(SubsamplingLevel& subsamplingLevel)
{
    subsamplingLevel = static_cast<SubsamplingLevel>(static_cast<int>(subsamplingLevel) + 1);
    ASSERT(subsamplingLevel <= SubsamplingLevel::Max);
    return subsamplingLevel;
}

typedef int RepetitionCount;

enum {
    RepetitionCountNone = 0,
    RepetitionCountOnce = 1,
    RepetitionCountInfinite = -1,
};

enum class AlphaOption {
    Premultiplied,
    NotPremultiplied
};

enum class GammaAndColorProfileOption {
    Applied,
    Ignored
};

enum class ImageAnimatingState : bool { No, Yes };

enum class EncodedDataStatus {
    Error,
    Unknown,
    TypeAvailable,
    SizeAvailable,
    Complete
};

enum class DecodingStatus {
    Invalid,
    Partial,
    Complete,
    Decoding
};

enum class ImageDrawResult {
    DidNothing,
    DidRequestDecoding,
    DidRecord,
    DidDraw
};

enum class ShowDebugBackground : bool {
    No,
    Yes
};

enum class AllowImageSubsampling : bool {
    No,
    Yes
};

struct Headroom {
    constexpr Headroom(float headroom)
    {
        this->headroom = headroom;
    }

    constexpr operator float() const { return headroom; }

    friend constexpr bool operator==(const Headroom&, const Headroom&) = default;

    static const Headroom FromImage;
    static const Headroom None;

    float headroom;
};

constexpr const Headroom Headroom::FromImage = { 0 };
constexpr const Headroom Headroom::None = { 1 };

#if USE(SKIA)
enum class StrictImageClamping : bool {
    No,
    Yes
};
#endif

}
