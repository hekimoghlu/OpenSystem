/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

#if ENABLE(GPU_PROCESS)

#include "MessageReceiver.h"
#include "RemoteAudioHardwareListenerIdentifier.h"
#include <WebCore/AudioHardwareListener.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class GPUConnectionToWebProcess;

class RemoteAudioHardwareListenerProxy final : private WebCore::AudioHardwareListener::Client {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioHardwareListenerProxy);
public:
    RemoteAudioHardwareListenerProxy(GPUConnectionToWebProcess&, RemoteAudioHardwareListenerIdentifier&&);
    virtual ~RemoteAudioHardwareListenerProxy();

private:
    // AudioHardwareListener::Client
    void audioHardwareDidBecomeActive() final;
    void audioHardwareDidBecomeInactive() final;
    void audioOutputDeviceChanged() final;

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnection;
    RemoteAudioHardwareListenerIdentifier m_identifier;
    Ref<WebCore::AudioHardwareListener> m_listener;
};

}

#endif
