/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

#if USE(LIBWEBRTC)

#include "CAAudioStreamDescription.h"
#include "RealtimeIncomingAudioSource.h"
#include "WebAudioBufferList.h"
#include <CoreAudio/CoreAudioTypes.h>

typedef const struct opaqueCMFormatDescription *CMFormatDescriptionRef;

namespace WebCore {

class RealtimeIncomingAudioSourceCocoa final : public RealtimeIncomingAudioSource {
public:
    static Ref<RealtimeIncomingAudioSourceCocoa> create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&&, String&&);

private:
    RealtimeIncomingAudioSourceCocoa(rtc::scoped_refptr<webrtc::AudioTrackInterface>&&, String&&);

    // RealtimeMediaSource API
    void startProducingData() final;
    void stopProducingData()  final;

    // webrtc::AudioTrackSinkInterface API
    void OnData(const void* audioData, int bitsPerSample, int sampleRate, size_t numberOfChannels, size_t numberOfFrames) final;

#if !RELEASE_LOG_DISABLED
    void logTimerFired();
#endif

    static constexpr Seconds LogTimerInterval = 2_s;
    static constexpr size_t ChunksReceivedCountForLogging = 200; // 200 chunks of 10ms = 2s.

    uint64_t m_numberOfFrames { 0 };

    int m_sampleRate { 0 };
    size_t m_numberOfChannels { 0 };
    CAAudioStreamDescription m_streamDescription;
    std::unique_ptr<WebAudioBufferList> m_audioBufferList;
    size_t m_chunksReceived { 0 };
#if !RELEASE_LOG_DISABLED
    size_t m_lastChunksReceived { 0 };
    bool m_audioFormatChanged { false };
    Timer m_logTimer;
#endif
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
