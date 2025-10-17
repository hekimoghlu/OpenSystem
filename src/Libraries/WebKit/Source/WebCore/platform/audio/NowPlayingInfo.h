/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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

#include "Image.h"
#include "MediaUniqueIdentifier.h"
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct NowPlayingInfoArtwork {
    String src;
    String mimeType;
    RefPtr<Image> image;

    bool operator==(const NowPlayingInfoArtwork& other) const
    {
        return src == other.src && mimeType == other.mimeType;
    }
};

struct NowPlayingMetadata {
    String title;
    String artist;
    String album;
    String sourceApplicationIdentifier;
    std::optional<NowPlayingInfoArtwork> artwork;

    friend bool operator==(const NowPlayingMetadata&, const NowPlayingMetadata&) = default;
};

struct NowPlayingInfo {
    NowPlayingMetadata metadata;
    double duration { 0 };
    double currentTime { 0 };
    double rate { 1.0 };
    bool supportsSeeking { false };
    Markable<MediaUniqueIdentifier> uniqueIdentifier;
    bool isPlaying { false };
    bool allowsNowPlayingControlsVisibility { false };
    bool isVideo { false };

    friend bool operator==(const NowPlayingInfo&, const NowPlayingInfo&) = default;
};

} // namespace WebCore
