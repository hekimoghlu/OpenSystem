/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#if ENABLE(WEB_RTC)

#include <span>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

struct RTCEncodedAudioFrameMetadata {
    uint32_t synchronizationSource;
    Vector<uint32_t> contributingSources;
};

struct RTCEncodedVideoFrameMetadata {
    std::optional<int64_t> frameId;
    Vector<int64_t> dependencies;
    uint16_t width;
    uint16_t height;
    std::optional<int32_t> spatialIndex;
    std::optional<int32_t> temporalIndex;
    uint32_t synchronizationSource;
};

class RTCRtpTransformableFrame : public RefCounted<RTCRtpTransformableFrame> {
public:
    virtual ~RTCRtpTransformableFrame() = default;

    virtual std::span<const uint8_t> data() const = 0;
    virtual void setData(std::span<const uint8_t>) = 0;

    virtual uint64_t timestamp() const = 0;
    virtual RTCEncodedAudioFrameMetadata audioMetadata() const = 0;
    virtual RTCEncodedVideoFrameMetadata videoMetadata() const = 0;

    virtual bool isKeyFrame() const = 0;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
