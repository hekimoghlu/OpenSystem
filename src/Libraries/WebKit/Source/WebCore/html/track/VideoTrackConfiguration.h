/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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

#include "PlatformVideoTrackConfiguration.h"
#include "VideoColorSpace.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using VideoTrackConfigurationInit = PlatformVideoTrackConfiguration;

class VideoTrackConfiguration : public RefCounted<VideoTrackConfiguration> {
    WTF_MAKE_TZONE_ALLOCATED(VideoTrackConfiguration);
public:
    static Ref<VideoTrackConfiguration> create(VideoTrackConfigurationInit&& init) { return adoptRef(*new VideoTrackConfiguration(WTFMove(init))); }
    static Ref<VideoTrackConfiguration> create() { return adoptRef(*new VideoTrackConfiguration()); }

    void setState(const VideoTrackConfigurationInit& state)
    {
        m_state = state;
        m_colorSpace->setState(m_state.colorSpace);
    }

    String codec() const { return m_state.codec; }
    void setCodec(String codec) { m_state.codec = codec; }

    uint32_t width() const { return m_state.width; }
    void setWidth(uint32_t width) { m_state.width = width; }

    uint32_t height() const { return m_state.height; }
    void setHeight(uint32_t height) { m_state.height = height; }

    Ref<VideoColorSpace> colorSpace() const { return m_colorSpace; }
    void setColorSpace(Ref<VideoColorSpace>&& colorSpace) { m_colorSpace = WTFMove(colorSpace); }

    double framerate() const { return m_state.framerate; }
    void setFramerate(double framerate) { m_state.framerate = framerate; }

    uint64_t bitrate() const { return m_state.bitrate; }
    void setBitrate(uint64_t bitrate) { m_state.bitrate = bitrate; }

    std::optional<SpatialVideoMetadata> spatialVideoMetadata() const { return m_state.spatialVideoMetadata; }
    void setSpatialVideoMetadata(const SpatialVideoMetadata& metadata) { m_state.spatialVideoMetadata = metadata; }

    Ref<JSON::Object> toJSON() const;

private:
    VideoTrackConfiguration(VideoTrackConfigurationInit&& init)
        : m_state(init)
        , m_colorSpace(VideoColorSpace::create(init.colorSpace))
    {
    }
    VideoTrackConfiguration()
        : m_colorSpace(VideoColorSpace::create())
    {
    }

    VideoTrackConfigurationInit m_state;
    Ref<VideoColorSpace> m_colorSpace;
};

}

#endif
