/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#include "MessageReceiver.h"
#include "MessageSender.h"
#include <WebCore/RealtimeMediaSourceIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

#if PLATFORM(COCOA)
#include "SharedCARingBuffer.h"
#endif

namespace WTF {
class MediaTime;
}

namespace WebCore {
class CaptureDevice;
}

namespace WebKit {

class SpeechRecognitionRemoteRealtimeMediaSource;
class WebProcessProxy;
struct SharedPreferencesForWebProcess;

class SpeechRecognitionRemoteRealtimeMediaSourceManager final : public IPC::MessageReceiver, public IPC::MessageSender {
    WTF_MAKE_TZONE_ALLOCATED(SpeechRecognitionRemoteRealtimeMediaSourceManager);
public:
    explicit SpeechRecognitionRemoteRealtimeMediaSourceManager(const WebProcessProxy&);

    void ref() const final;
    void deref() const final;

    void addSource(SpeechRecognitionRemoteRealtimeMediaSource&, const WebCore::CaptureDevice&);
    void removeSource(SpeechRecognitionRemoteRealtimeMediaSource&);

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    // Messages::SpeechRecognitionRemoteRealtimeMediaSourceManager
    void remoteAudioSamplesAvailable(WebCore::RealtimeMediaSourceIdentifier, const WTF::MediaTime&, uint64_t numberOfFrames);
    void remoteCaptureFailed(WebCore::RealtimeMediaSourceIdentifier);
    void remoteSourceStopped(WebCore::RealtimeMediaSourceIdentifier);
#if PLATFORM(COCOA)
    void setStorage(WebCore::RealtimeMediaSourceIdentifier, ConsumerSharedCARingBuffer::Handle&&, const WebCore::CAAudioStreamDescription&);
#endif

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    WeakRef<const WebProcessProxy> m_process;
    HashMap<WebCore::RealtimeMediaSourceIdentifier, ThreadSafeWeakPtr<SpeechRecognitionRemoteRealtimeMediaSource>> m_sources;
};

} // namespace WebKit

#endif
