/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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

#if ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO)

#include "Connection.h"
#include "IPCSemaphore.h"
#include "RemoteAudioDestinationIdentifier.h"
#include <WebCore/SharedMemory.h>
#include <memory>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakRef.h>

#if PLATFORM(COCOA)
#include "SharedCARingBuffer.h"
#endif

namespace WebCore {
#if PLATFORM(COCOA)
class CAAudioStreamDescription;
#endif
class SharedMemoryHandle;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteAudioDestination;
struct SharedPreferencesForWebProcess;

class RemoteAudioDestinationManager : private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioDestinationManager);
    WTF_MAKE_NONCOPYABLE(RemoteAudioDestinationManager);
public:
    RemoteAudioDestinationManager(GPUConnectionToWebProcess&);
    ~RemoteAudioDestinationManager();

    void ref() const final;
    void deref() const final;

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }

    bool allowsExitUnderMemoryPressure() const;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    void createAudioDestination(RemoteAudioDestinationIdentifier, const String& inputDeviceId, uint32_t numberOfInputChannels, uint32_t numberOfOutputChannels, float sampleRate, float hardwareSampleRate, IPC::Semaphore&& renderSemaphore, WebCore::SharedMemoryHandle&&, CompletionHandler<void(size_t)>&&);
    void deleteAudioDestination(RemoteAudioDestinationIdentifier);
    void startAudioDestination(RemoteAudioDestinationIdentifier, CompletionHandler<void(bool, size_t)>&&);
    void stopAudioDestination(RemoteAudioDestinationIdentifier, CompletionHandler<void(bool)>&&);
#if PLATFORM(COCOA)
    void audioSamplesStorageChanged(RemoteAudioDestinationIdentifier, ConsumerSharedCARingBuffer::Handle&&);
#endif

    HashMap<RemoteAudioDestinationIdentifier, UniqueRef<RemoteAudioDestination>> m_audioDestinations;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
};

} // namespace WebKit;

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO)
