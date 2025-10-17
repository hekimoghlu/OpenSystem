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

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MessageReceiver.h"
#include "RemoteCDMIdentifier.h"
#include "RemoteCDMInstanceIdentifier.h"
#include "RemoteCDMInstanceSessionIdentifier.h"
#include <WebCore/CDMPrivate.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class RemoteCDMInstanceProxy;
class RemoteCDMInstanceSessionProxy;
class RemoteCDMProxy;
struct RemoteCDMConfiguration;

class RemoteCDMFactoryProxy final : public RefCounted<RemoteCDMFactoryProxy>, public IPC::MessageReceiver, WebCore::CDMPrivateClient {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCDMFactoryProxy);
public:
    static Ref<RemoteCDMFactoryProxy> create(GPUConnectionToWebProcess& gpuConnection)
    {
        return adoptRef(*new RemoteCDMFactoryProxy(gpuConnection));
    }
    virtual ~RemoteCDMFactoryProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void clear();

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }
    bool didReceiveSyncMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder, UniqueRef<IPC::Encoder>& encoder) { return didReceiveSyncMessage(connection, decoder, encoder); }
    void didReceiveCDMMessage(IPC::Connection&, IPC::Decoder&);
    void didReceiveCDMInstanceMessage(IPC::Connection&, IPC::Decoder&);
    void didReceiveCDMInstanceSessionMessage(IPC::Connection&, IPC::Decoder&);
    bool didReceiveSyncCDMMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);
    bool didReceiveSyncCDMInstanceMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);
    bool didReceiveSyncCDMInstanceSessionMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&);

    void addProxy(const RemoteCDMIdentifier&, RefPtr<RemoteCDMProxy>&&);
    void removeProxy(const RemoteCDMIdentifier&);

    void addInstance(const RemoteCDMInstanceIdentifier&, Ref<RemoteCDMInstanceProxy>&&);
    void removeInstance(const RemoteCDMInstanceIdentifier&);
    RemoteCDMInstanceProxy* getInstance(const RemoteCDMInstanceIdentifier&);

    void addSession(const RemoteCDMInstanceSessionIdentifier&, Ref<RemoteCDMInstanceSessionProxy>&&);
    void removeSession(const RemoteCDMInstanceSessionIdentifier&);

    RefPtr<GPUConnectionToWebProcess> gpuConnectionToWebProcess() { return m_gpuConnectionToWebProcess.get(); }

    bool allowsExitUnderMemoryPressure() const;

    const String& mediaKeysStorageDirectory() const;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const;
#endif

private:
    RemoteCDMFactoryProxy(GPUConnectionToWebProcess&);

    friend class GPUProcessConnection;
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    // Messages
    void createCDM(const String& keySystem, CompletionHandler<void(std::optional<RemoteCDMIdentifier>&&, RemoteCDMConfiguration&&)>&&);
    void supportsKeySystem(const String& keySystem, CompletionHandler<void(bool)>&&);

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    HashMap<RemoteCDMIdentifier, RefPtr<RemoteCDMProxy>> m_proxies;
    HashMap<RemoteCDMInstanceIdentifier, Ref<RemoteCDMInstanceProxy>> m_instances;
    HashMap<RemoteCDMInstanceSessionIdentifier, Ref<RemoteCDMInstanceSessionProxy>> m_sessions;

#if !RELEASE_LOG_DISABLED
    mutable RefPtr<Logger> m_logger;
#endif
};

}

#endif
