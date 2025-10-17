/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

#include "LibWebRTCMacros.h"
#include "RealtimeMediaSource.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/media_stream_interface.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

#include <wtf/RetainPtr.h>

namespace WebCore {

class LibWebRTCAudioModule;

class RealtimeIncomingAudioSource
    : public RealtimeMediaSource
    , private webrtc::AudioTrackSinkInterface
    , private webrtc::ObserverInterface
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RealtimeIncomingAudioSource, WTF::DestructionThread::MainRunLoop>
{
public:
    static Ref<RealtimeIncomingAudioSource> create(rtc::scoped_refptr<webrtc::AudioTrackInterface>&&, String&&);

    void setAudioModule(RefPtr<LibWebRTCAudioModule>&&);
    LibWebRTCAudioModule* audioModule() { return m_audioModule.get(); }

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;
    ~RealtimeIncomingAudioSource();
protected:
    RealtimeIncomingAudioSource(rtc::scoped_refptr<webrtc::AudioTrackInterface>&&, String&&);

#if !RELEASE_LOG_DISABLED
    ASCIILiteral logClassName() const final { return "RealtimeIncomingAudioSource"_s; }
#endif

    // RealtimeMediaSource API
    void startProducingData() override;
    void stopProducingData()  override;

private:
    // webrtc::AudioTrackSinkInterface API
    void OnData(const void* /* audioData */, int /* bitsPerSample */, int /* sampleRate */, size_t /* numberOfChannels */, size_t /* numberOfFrames */) override { };

    // webrtc::ObserverInterface API
    void OnChanged() final;

    const RealtimeMediaSourceCapabilities& capabilities() final;
    const RealtimeMediaSourceSettings& settings() final;

    bool isIncomingAudioSource() const final { return true; }

    RealtimeMediaSourceSettings m_currentSettings;
    rtc::scoped_refptr<webrtc::AudioTrackInterface> m_audioTrack;
    RefPtr<LibWebRTCAudioModule> m_audioModule;

#if !RELEASE_LOG_DISABLED
    mutable RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RealtimeIncomingAudioSource)
    static bool isType(const WebCore::RealtimeMediaSource& source) { return source.isIncomingAudioSource(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // USE(LIBWEBRTC)
