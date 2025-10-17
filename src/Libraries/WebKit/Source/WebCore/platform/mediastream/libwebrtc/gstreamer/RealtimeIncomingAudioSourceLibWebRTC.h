/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

#if USE(LIBWEBRTC) && USE(GSTREAMER)

#include "RealtimeIncomingAudioSource.h"

#include <wtf/MediaTime.h>

namespace WebCore {

class RealtimeIncomingAudioSourceLibWebRTC final : public RealtimeIncomingAudioSource {
public:
    static Ref<RealtimeIncomingAudioSourceLibWebRTC> create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&&, String&&);

private:
    RealtimeIncomingAudioSourceLibWebRTC(rtc::scoped_refptr<webrtc::AudioTrackInterface>&&, String&&);

    // webrtc::AudioTrackSinkInterface API
    void OnData(const void* audioData, int bitsPerSample, int sampleRate, size_t numberOfChannels, size_t numberOfFrames) final;

    uint64_t m_numberOfFrames { 0 };
    MediaTime m_baseTime { MediaTime::invalidTime() };
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)

