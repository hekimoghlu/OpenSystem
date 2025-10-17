/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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

#include "VideoFrame.h"
#include "VideoPixelFormat.h"

namespace WebCore {

struct WebCodecsVideoFrameData {
    // We might want to make memory cost take into account the video frame format.
    size_t memoryCost() const { return 4 * codedWidth * codedHeight; }

    RefPtr<VideoFrame> internalFrame;
    std::optional<VideoPixelFormat> format;
    size_t codedWidth { 0 };
    size_t codedHeight { 0 };
    size_t displayWidth { 0 };
    size_t displayHeight { 0 };
    size_t visibleWidth { 0 };
    size_t visibleHeight { 0 };
    size_t visibleLeft { 0 };
    size_t visibleTop { 0 };
    std::optional<uint64_t> duration { 0 };
    int64_t timestamp { 0 };
};

}

#endif
