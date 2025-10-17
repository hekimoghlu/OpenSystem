/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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

#include "MessageReceiver.h"
#include "RemoteCDMFactory.h"
#include "RemoteCDMInstanceConfiguration.h"
#include "RemoteCDMInstanceIdentifier.h"
#include <WebCore/CDMInstance.h>

namespace WebKit {

class RemoteCDMInstance final : public WebCore::CDMInstance, private IPC::MessageReceiver {
public:
    virtual ~RemoteCDMInstance();
    static Ref<RemoteCDMInstance> create(WeakPtr<RemoteCDMFactory>&&, RemoteCDMInstanceIdentifier&&, RemoteCDMInstanceConfiguration&&);

    void ref() const final { WebCore::CDMInstance::ref(); }
    void deref() const final { WebCore::CDMInstance::deref(); }

    const RemoteCDMInstanceIdentifier& identifier() const { return m_identifier; }

private:
    RemoteCDMInstance(WeakPtr<RemoteCDMFactory>&&, RemoteCDMInstanceIdentifier&&, RemoteCDMInstanceConfiguration&&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void unrequestedInitializationDataReceived(const String&, Ref<WebCore::SharedBuffer>&&);

    ImplementationType implementationType() const final { return ImplementationType::Remote; }
    void initializeWithConfiguration(const WebCore::CDMKeySystemConfiguration&, AllowDistinctiveIdentifiers, AllowPersistentState, SuccessCallback&&) final;
    void setServerCertificate(Ref<WebCore::SharedBuffer>&&, SuccessCallback&&) final;
    void setStorageDirectory(const String&) final;
    const String& keySystem() const final { return m_configuration.keySystem; }
    RefPtr<WebCore::CDMInstanceSession> createSession() final;
    void setClient(WeakPtr<WebCore::CDMInstanceClient>&& client) final { m_client = WTFMove(client); }
    void clearClient() final { m_client.clear(); }

    WeakPtr<RemoteCDMFactory> m_factory;
    RemoteCDMInstanceIdentifier m_identifier;
    RemoteCDMInstanceConfiguration m_configuration;
    WeakPtr<WebCore::CDMInstanceClient> m_client;
};

}

SPECIALIZE_TYPE_TRAITS_CDM_INSTANCE(WebKit::RemoteCDMInstance, WebCore::CDMInstance::ImplementationType::Remote)

#endif
