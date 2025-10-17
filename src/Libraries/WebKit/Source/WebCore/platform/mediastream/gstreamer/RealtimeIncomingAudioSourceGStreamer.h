/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#include "RealtimeIncomingSourceGStreamer.h"

namespace WebCore {

class RealtimeIncomingAudioSourceGStreamer : public RealtimeIncomingSourceGStreamer {
public:
    static Ref<RealtimeIncomingAudioSourceGStreamer> create(AtomString&& audioTrackId) { return adoptRef(*new RealtimeIncomingAudioSourceGStreamer(WTFMove(audioTrackId))); }

protected:
    RealtimeIncomingAudioSourceGStreamer(AtomString&&);
    ~RealtimeIncomingAudioSourceGStreamer();

private:
    // RealtimeMediaSource API
    const RealtimeMediaSourceSettings& settings() final;
    bool isIncomingAudioSource() const final { return true; }

    // RealtimeIncomingSourceGStreamer API
    void dispatchSample(GRefPtr<GstSample>&&) final;

    RealtimeMediaSourceSettings m_currentSettings;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RealtimeIncomingAudioSourceGStreamer)
    static bool isType(const WebCore::RealtimeMediaSource& source) { return source.isIncomingAudioSource(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(GSTREAMER_WEBRTC)
