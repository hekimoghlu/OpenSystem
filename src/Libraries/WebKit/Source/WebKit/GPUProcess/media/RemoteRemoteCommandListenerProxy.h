/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#include "RemoteRemoteCommandListenerIdentifier.h"
#include <WebCore/RemoteCommandListener.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class GPUConnectionToWebProcess;
struct SharedPreferencesForWebProcess;

class RemoteRemoteCommandListenerProxy : public RefCounted<RemoteRemoteCommandListenerProxy>, private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteRemoteCommandListenerProxy);
public:
    static Ref<RemoteRemoteCommandListenerProxy> create(GPUConnectionToWebProcess& process, RemoteRemoteCommandListenerIdentifier&& identifier)
    {
        return adoptRef(*new RemoteRemoteCommandListenerProxy(process, WTFMove(identifier)));
    }

    virtual ~RemoteRemoteCommandListenerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    bool supportsSeeking() const { return m_supportsSeeking; }
    const WebCore::RemoteCommandListener::RemoteCommandsSet& supportedCommands() const { return m_supportedCommands; }

    RemoteRemoteCommandListenerIdentifier identifier() const { return m_identifier; }

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    RemoteRemoteCommandListenerProxy(GPUConnectionToWebProcess&, RemoteRemoteCommandListenerIdentifier&&);

    // Messages
    void updateSupportedCommands(Vector<WebCore::PlatformMediaSession::RemoteControlCommandType>&& commands, bool supportsSeeking);

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnection;
    RemoteRemoteCommandListenerIdentifier m_identifier;
    WebCore::RemoteCommandListener::RemoteCommandsSet m_supportedCommands;
    bool m_supportsSeeking { false };
};

}

#endif
