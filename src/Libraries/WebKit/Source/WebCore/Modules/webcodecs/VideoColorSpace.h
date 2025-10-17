/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

#if ENABLE(VIDEO)

#include "VideoColorPrimaries.h"
#include "VideoColorSpaceInit.h"
#include "VideoMatrixCoefficients.h"
#include "VideoTransferCharacteristics.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class VideoColorSpace : public RefCounted<VideoColorSpace> {
    WTF_MAKE_TZONE_ALLOCATED(VideoColorSpace);
public:
    static Ref<VideoColorSpace> create() { return adoptRef(*new VideoColorSpace()); };
    static Ref<VideoColorSpace> create(const VideoColorSpaceInit& init) { return adoptRef(*new VideoColorSpace(init)); }
    static Ref<VideoColorSpace> create(VideoColorSpaceInit&& init) { return adoptRef(*new VideoColorSpace(WTFMove(init))); }

    void setState(const VideoColorSpaceInit& state) { m_state = state; }

    const std::optional<VideoColorPrimaries>& primaries() const { return m_state.primaries; }
    void setPrimaries(std::optional<VideoColorPrimaries>&& primaries) { m_state.primaries = WTFMove(primaries); }

    const std::optional<VideoTransferCharacteristics>& transfer() const { return m_state.transfer; }
    void setTransfer(std::optional<VideoTransferCharacteristics>&& transfer) { m_state.transfer = WTFMove(transfer); }

    const std::optional<VideoMatrixCoefficients>& matrix() const { return m_state.matrix; }
    void setMatrix(std::optional<VideoMatrixCoefficients>&& matrix) { m_state.matrix = WTFMove(matrix); }

    const std::optional<bool>& fullRange() const { return m_state.fullRange; }
    void setfFullRange(std::optional<bool>&& fullRange) { m_state.fullRange = WTFMove(fullRange); }

    VideoColorSpaceInit state() const { return m_state; }

    Ref<JSON::Object> toJSON() const;

private:
    VideoColorSpace() = default;
    VideoColorSpace(const VideoColorSpaceInit& init)
        : m_state(init)
    {
    }
    VideoColorSpace(VideoColorSpaceInit&& init)
        : m_state(WTFMove(init))
    {
    }

    VideoColorSpaceInit m_state { };
};

}

#endif
