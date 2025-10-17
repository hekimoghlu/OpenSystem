/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MessageReceiver.h"
#include "RemoteLegacyCDMIdentifier.h"
#include "RemoteLegacyCDMSessionIdentifier.h"
#include <WebCore/MediaPlayerIdentifier.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class RemoteLegacyCDMSessionProxy;
class RemoteLegacyCDMProxy;
struct RemoteLegacyCDMConfiguration;

class RemoteLegacyCDMFactoryProxy final : public RefCounted<RemoteLegacyCDMFactoryProxy>, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLegacyCDMFactoryProxy);
public:
    static Ref<RemoteLegacyCDMFactoryProxy> create(GPUConnectionToWebProcess& gpuConnectionToWebProcess)
    {
        return adoptRef(*new RemoteLegacyCDMFactoryProxy(gpuConnectionToWebProcess));
    }

    virtual ~RemoteLegacyCDMFactoryProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void clear();

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }
    bool didReceiveSyncMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& encoder) { return didReceiveSyncMessage(connection, decoder, encoder); }
    void didReceiveCDMMessage(IPC::Connection&, IPC::Decoder&);
    void didReceiveCDMSessionMessage(IPC::Connection&, IPC::Decoder&);
    bool didReceiveSyncCDMMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);
    bool didReceiveSyncCDMSessionMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    void addProxy(RemoteLegacyCDMIdentifier, Ref<RemoteLegacyCDMProxy>&&);
    void removeProxy(RemoteLegacyCDMIdentifier);

    void addSession(RemoteLegacyCDMSessionIdentifier, Ref<RemoteLegacyCDMSessionProxy>&&);
    void removeSession(RemoteLegacyCDMSessionIdentifier, CompletionHandler<void()>&&);
    RemoteLegacyCDMSessionProxy* getSession(const RemoteLegacyCDMSessionIdentifier&) const;

    RefPtr<GPUConnectionToWebProcess> gpuConnectionToWebProcess() { return m_gpuConnectionToWebProcess.get(); }

    bool allowsExitUnderMemoryPressure() const;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const;
#endif

private:
    RemoteLegacyCDMFactoryProxy(GPUConnectionToWebProcess&);

    friend class GPUProcessConnection;
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    // Messages
    void createCDM(const String& keySystem, std::optional<WebCore::MediaPlayerIdentifier>&&, CompletionHandler<void(std::optional<RemoteLegacyCDMIdentifier>&&)>&&);
    void supportsKeySystem(const String& keySystem, std::optional<String> mimeType, CompletionHandler<void(bool)>&&);

    WeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    HashMap<RemoteLegacyCDMIdentifier, Ref<RemoteLegacyCDMProxy>> m_proxies;
    HashMap<RemoteLegacyCDMSessionIdentifier, Ref<RemoteLegacyCDMSessionProxy>> m_sessions;

#if !RELEASE_LOG_DISABLED
    mutable RefPtr<Logger> m_logger;
#endif
};

}

#endif
