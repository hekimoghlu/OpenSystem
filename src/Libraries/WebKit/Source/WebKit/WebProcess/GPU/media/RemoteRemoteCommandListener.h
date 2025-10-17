/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
#include "RemoteRemoteCommandListenerIdentifier.h"
#include <WebCore/RemoteCommandListener.h>
#include <wtf/Identified.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class GPUProcessConnection;
class WebProcess;

class RemoteRemoteCommandListener final
    : public WebCore::RemoteCommandListener
    , private Identified<RemoteRemoteCommandListenerIdentifier>
    , private GPUProcessConnection::Client
    , private IPC::MessageReceiver
    , public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteRemoteCommandListener> {
public:
    static Ref<RemoteRemoteCommandListener> create(WebCore::RemoteCommandListenerClient&);
    explicit RemoteRemoteCommandListener(WebCore::RemoteCommandListenerClient&);
    ~RemoteRemoteCommandListener();

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    // Messages
    void didReceiveRemoteControlCommand(WebCore::PlatformMediaSession::RemoteControlCommandType, const WebCore::PlatformMediaSession::RemoteCommandArgument&);

    // WebCore::RemoteCommandListener
    void updateSupportedCommands() final;

    GPUProcessConnection& ensureGPUProcessConnection();

    WebCore::RemoteCommandListener::RemoteCommandsSet m_currentCommands;
    bool m_currentSupportSeeking { false };
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
};

}

#endif
