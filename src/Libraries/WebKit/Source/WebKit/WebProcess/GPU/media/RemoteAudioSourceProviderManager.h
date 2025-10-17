/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS)

#include "Connection.h"
#include "SharedCARingBuffer.h"
#include "WebProcess.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/CAAudioStreamDescription.h>
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebCore/SharedMemory.h>
#include <WebCore/WebAudioBufferList.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class RemoteAudioSourceProvider;

class RemoteAudioSourceProviderManager : public IPC::WorkQueueMessageReceiver {
public:
    static Ref<RemoteAudioSourceProviderManager> create() { return adoptRef(*new RemoteAudioSourceProviderManager()); }
    ~RemoteAudioSourceProviderManager();
    void stopListeningForIPC();

    void addProvider(Ref<RemoteAudioSourceProvider>&&);
    void removeProvider(WebCore::MediaPlayerIdentifier);

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

private:
    RemoteAudioSourceProviderManager();

    // Messages
    void audioStorageChanged(WebCore::MediaPlayerIdentifier, ConsumerSharedCARingBuffer::Handle&&, const WebCore::CAAudioStreamDescription&);
    void audioSamplesAvailable(WebCore::MediaPlayerIdentifier, uint64_t startFrame, uint64_t numberOfFrames);

    void setConnection(IPC::Connection*);

    class RemoteAudio {
        WTF_MAKE_TZONE_ALLOCATED(RemoteAudio);
    public:
        explicit RemoteAudio(Ref<RemoteAudioSourceProvider>&&);

        void setStorage(ConsumerSharedCARingBuffer::Handle&&, const WebCore::CAAudioStreamDescription&);
        void audioSamplesAvailable(uint64_t startFrame, uint64_t numberOfFrames);

    private:
        Ref<RemoteAudioSourceProvider> m_provider;
        std::optional<WebCore::CAAudioStreamDescription> m_description;
        std::unique_ptr<ConsumerSharedCARingBuffer> m_ringBuffer;
        std::unique_ptr<WebCore::WebAudioBufferList> m_buffer;
    };

    Ref<WorkQueue> m_queue;
    RefPtr<IPC::Connection> m_connection;

    // background thread member
    HashMap<WebCore::MediaPlayerIdentifier, std::unique_ptr<RemoteAudio>> m_providers;
};

} // namespace WebKit

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS)
