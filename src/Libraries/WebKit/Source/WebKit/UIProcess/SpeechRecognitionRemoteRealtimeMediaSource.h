/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

#include <WebCore/RealtimeMediaSource.h>
#include <WebCore/RealtimeMediaSourceIdentifier.h>
#include <wtf/MediaTime.h>

#if PLATFORM(COCOA)
#include "SharedCARingBuffer.h"
#include <WebCore/CAAudioStreamDescription.h>
#endif

namespace WebCore {
class CaptureDevice;
#if PLATFORM(COCOA)
class WebAudioBufferList;
#endif
}

namespace WebKit {
class SpeechRecognitionRemoteRealtimeMediaSourceManager;
    
class SpeechRecognitionRemoteRealtimeMediaSource : public WebCore::RealtimeMediaSource, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SpeechRecognitionRemoteRealtimeMediaSource, WTF::DestructionThread::MainRunLoop> {
public:
    static Ref<WebCore::RealtimeMediaSource> create(SpeechRecognitionRemoteRealtimeMediaSourceManager&, const WebCore::CaptureDevice&, WebCore::PageIdentifier);
    ~SpeechRecognitionRemoteRealtimeMediaSource();
    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

    WebCore::RealtimeMediaSourceIdentifier identifier() const { return m_identifier; }

#if PLATFORM(COCOA)
    void setStorage(ConsumerSharedCARingBuffer::Handle&&, const WebCore::CAAudioStreamDescription&);
#endif

    void remoteAudioSamplesAvailable(WTF::MediaTime, uint64_t numberOfFrames);
    void remoteCaptureFailed();
    void remoteSourceStopped();

private:
    SpeechRecognitionRemoteRealtimeMediaSource(WebCore::RealtimeMediaSourceIdentifier, SpeechRecognitionRemoteRealtimeMediaSourceManager&, const WebCore::CaptureDevice&, WebCore::PageIdentifier);

    // WebCore::RealtimeMediaSource
    void startProducingData() final;
    void stopProducingData() final;
    const WebCore::RealtimeMediaSourceCapabilities& capabilities() final { return m_capabilities; }
    const WebCore::RealtimeMediaSourceSettings& settings() final { return m_settings; }

    WebCore::RealtimeMediaSourceIdentifier m_identifier;
    WeakPtr<SpeechRecognitionRemoteRealtimeMediaSourceManager> m_manager;
    WebCore::RealtimeMediaSourceCapabilities m_capabilities;
    WebCore::RealtimeMediaSourceSettings m_settings;

#if PLATFORM(COCOA)
    std::optional<WebCore::CAAudioStreamDescription> m_description;
    std::unique_ptr<ConsumerSharedCARingBuffer> m_ringBuffer;
    std::unique_ptr<WebCore::WebAudioBufferList> m_buffer;
#endif
};

} // namespace WebKit

#endif
