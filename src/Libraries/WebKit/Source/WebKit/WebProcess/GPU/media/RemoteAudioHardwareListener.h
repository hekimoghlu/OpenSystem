/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include "RemoteAudioHardwareListenerIdentifier.h"
#include <WebCore/AudioHardwareListener.h>
#include <wtf/Identified.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class GPUProcessConnection;
class WebProcess;

class RemoteAudioHardwareListener final
    : public WebCore::AudioHardwareListener
    , private Identified<RemoteAudioHardwareListenerIdentifier>
    , private GPUProcessConnection::Client
    , private IPC::MessageReceiver
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteAudioHardwareListener> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteAudioHardwareListener);
public:
    static Ref<RemoteAudioHardwareListener> create(WebCore::AudioHardwareListener::Client&);
    ~RemoteAudioHardwareListener();

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    explicit RemoteAudioHardwareListener(WebCore::AudioHardwareListener::Client&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    // Messages
    void audioHardwareDidBecomeActive();
    void audioHardwareDidBecomeInactive();
    void audioOutputDeviceChanged(size_t bufferSizeMinimum, size_t bufferSizeMaximum);

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
};

}

#endif
