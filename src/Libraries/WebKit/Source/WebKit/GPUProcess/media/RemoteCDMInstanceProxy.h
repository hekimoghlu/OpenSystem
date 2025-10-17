/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#include "MessageReceiver.h"
#include "RemoteCDMInstanceIdentifier.h"
#include "RemoteCDMInstanceSessionIdentifier.h"
#include "RemoteCDMProxy.h"
#include <WebCore/CDMInstance.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
class CDMInstance;
struct CDMKeySystemConfiguration;
}

namespace IPC {
class SharedBufferReference;
}

namespace WebKit {

struct RemoteCDMInstanceConfiguration;
class RemoteCDMInstanceSessionProxy;

class RemoteCDMInstanceProxy : public WebCore::CDMInstanceClient, private IPC::MessageReceiver, public RefCounted<RemoteCDMInstanceProxy>  {
public:
    USING_CAN_MAKE_WEAKPTR(WebCore::CDMInstanceClient);

    static Ref<RemoteCDMInstanceProxy> create(RemoteCDMProxy&, Ref<WebCore::CDMInstance>&&, RemoteCDMInstanceIdentifier);
    ~RemoteCDMInstanceProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    const RemoteCDMInstanceConfiguration& configuration() const { return m_configuration.get(); }
    WebCore::CDMInstance& instance() { return m_instance; }
    Ref<WebCore::CDMInstance> protectedInstance() const;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    friend class RemoteCDMFactoryProxy;
    RemoteCDMInstanceProxy(RemoteCDMProxy&, Ref<WebCore::CDMInstance>&&, UniqueRef<RemoteCDMInstanceConfiguration>&&, RemoteCDMInstanceIdentifier);

    // CDMInstanceClient
    void unrequestedInitializationDataReceived(const String&, Ref<WebCore::SharedBuffer>&&) final;
#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
#endif

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    using SuccessValue = WebCore::CDMInstance::SuccessValue;
    using AllowDistinctiveIdentifiers = WebCore::CDMInstance::AllowDistinctiveIdentifiers;
    using AllowPersistentState = WebCore::CDMInstance::AllowPersistentState;

    // Messages
    void initializeWithConfiguration(const WebCore::CDMKeySystemConfiguration&, AllowDistinctiveIdentifiers, AllowPersistentState, CompletionHandler<void(SuccessValue)>&&);
    void setServerCertificate(Ref<WebCore::SharedBuffer>&&, CompletionHandler<void(SuccessValue)>&&);
    void setStorageDirectory(const String&);
    void createSession(uint64_t logIdentifier, CompletionHandler<void(std::optional<RemoteCDMInstanceSessionIdentifier>)>&&);

    RefPtr<RemoteCDMProxy> protectedCdm() const;

    WeakPtr<RemoteCDMProxy> m_cdm;
    Ref<WebCore::CDMInstance> m_instance;
    UniqueRef<RemoteCDMInstanceConfiguration> m_configuration;
    RemoteCDMInstanceIdentifier m_identifier;
    HashMap<RemoteCDMInstanceSessionIdentifier, Ref<RemoteCDMInstanceSessionProxy>> m_sessions;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

}

#endif
