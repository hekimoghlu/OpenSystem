/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 10, 2023.
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

#if ENABLE(MEDIA_STREAM)

namespace WebCore {

struct MediaTrackSupportedConstraints {
    bool width { true };
    bool height { true };
    bool aspectRatio { true };
    bool frameRate { true };
    bool facingMode { true };
    bool volume { true };
    bool sampleRate { true };
    bool sampleSize { true };
    bool echoCancellation { true };
    bool deviceId { true };
    bool groupId { true };
    bool displaySurface { true };
    bool whiteBalanceMode { true };
    bool zoom { true };
    bool torch { true };
    bool backgroundBlur { true };
    bool powerEfficient { true };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
