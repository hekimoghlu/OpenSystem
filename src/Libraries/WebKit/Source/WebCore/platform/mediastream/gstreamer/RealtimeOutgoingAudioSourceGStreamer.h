/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

#include "RealtimeOutgoingMediaSourceGStreamer.h"

namespace WebCore {

class RealtimeOutgoingAudioSourceGStreamer final : public RealtimeOutgoingMediaSourceGStreamer {
public:
    static Ref<RealtimeOutgoingAudioSourceGStreamer> create(const RefPtr<UniqueSSRCGenerator>& ssrcGenerator, const String& mediaStreamId, MediaStreamTrack& track)
    {
        return adoptRef(*new RealtimeOutgoingAudioSourceGStreamer(ssrcGenerator, mediaStreamId, track));
    }
    static Ref<RealtimeOutgoingAudioSourceGStreamer> createMuted(const RefPtr<UniqueSSRCGenerator>& ssrcGenerator)
    {
        return adoptRef(*new RealtimeOutgoingAudioSourceGStreamer(ssrcGenerator));
    }
    ~RealtimeOutgoingAudioSourceGStreamer();

    WARN_UNUSED_RETURN GRefPtr<GstPad> outgoingSourcePad() const final;
    RefPtr<GStreamerRTPPacketizer> createPacketizer(RefPtr<UniqueSSRCGenerator>, const GstStructure*, GUniquePtr<GstStructure>&&) final;

    void dispatchBitrateRequest(uint32_t bitrate) final;

protected:
    explicit RealtimeOutgoingAudioSourceGStreamer(const RefPtr<UniqueSSRCGenerator>&, const String& mediaStreamId, MediaStreamTrack&);
    explicit RealtimeOutgoingAudioSourceGStreamer(const RefPtr<UniqueSSRCGenerator>&);

private:
    void initialize();

    RTCRtpCapabilities rtpCapabilities() const final;
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
