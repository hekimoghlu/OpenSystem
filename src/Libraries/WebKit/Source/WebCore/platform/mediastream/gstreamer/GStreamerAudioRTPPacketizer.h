/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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

#if USE(GSTREAMER_WEBRTC)

#include "GStreamerRTPPacketizer.h"
#include "GStreamerWebRTCUtils.h"

namespace WebCore {

class GStreamerAudioRTPPacketizer final : public GStreamerRTPPacketizer {
public:
    static RefPtr<GStreamerAudioRTPPacketizer> create(RefPtr<UniqueSSRCGenerator>, const GstStructure* codecParameters, GUniquePtr<GstStructure>&& encodingParameters);

private:
    explicit GStreamerAudioRTPPacketizer(GRefPtr<GstCaps>&& inputCaps, GRefPtr<GstElement>&& encoder, GRefPtr<GstElement>&& payloader, GUniquePtr<GstStructure>&& encodingParameters, GRefPtr<GstCaps>&& rtpCaps, std::optional<int>&& payloadType);

    GRefPtr<GstElement> m_audioconvert;
    GRefPtr<GstElement> m_audioresample;
    GRefPtr<GstElement> m_inputCapsFilter;
    GRefPtr<GstCaps> m_inputCaps;
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
