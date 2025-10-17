/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM) && PLATFORM(COCOA)

#include "AudioMediaStreamTrackRendererInternalUnitIdentifier.h"
#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "SharedCARingBuffer.h"
#include "SharedPreferencesForWebProcess.h"
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace IPC {
class Semaphore;
}

namespace WebCore {
class CAAudioStreamDescription;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteAudioDestination;
class RemoteAudioMediaStreamTrackRendererInternalUnitManagerUnit;

class RemoteAudioMediaStreamTrackRendererInternalUnitManager : private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioMediaStreamTrackRendererInternalUnitManager);
    WTF_MAKE_NONCOPYABLE(RemoteAudioMediaStreamTrackRendererInternalUnitManager);
public:
    explicit RemoteAudioMediaStreamTrackRendererInternalUnitManager(GPUConnectionToWebProcess&);
    ~RemoteAudioMediaStreamTrackRendererInternalUnitManager();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    bool hasUnits() { return !m_units.isEmpty(); }

    void notifyLastToCaptureAudioChanged();

    void ref() const final;
    void deref() const final;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    // Messages
    void createUnit(AudioMediaStreamTrackRendererInternalUnitIdentifier, const String&, CompletionHandler<void(std::optional<WebCore::CAAudioStreamDescription>, size_t)>&& callback);
    void deleteUnit(AudioMediaStreamTrackRendererInternalUnitIdentifier);
    void startUnit(AudioMediaStreamTrackRendererInternalUnitIdentifier, ConsumerSharedCARingBuffer::Handle&&, IPC::Semaphore&&);
    void stopUnit(AudioMediaStreamTrackRendererInternalUnitIdentifier);
    void setLastDeviceUsed(const String&);

    HashMap<AudioMediaStreamTrackRendererInternalUnitIdentifier, Ref<class RemoteAudioMediaStreamTrackRendererInternalUnitManagerUnit>> m_units;
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
};

} // namespace WebKit;

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM) && PLATFORM(COCOA)
