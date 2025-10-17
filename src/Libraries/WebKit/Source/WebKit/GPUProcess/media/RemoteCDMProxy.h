/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#include "RemoteCDMFactoryProxy.h"
#include "RemoteCDMInstanceIdentifier.h"
#include <WebCore/CDMPrivate.h>
#include <wtf/Forward.h>
#include <wtf/UniqueRef.h>

namespace WebCore {
class SharedBuffer;
enum class CDMRequirement : uint8_t;
enum class CDMSessionType : uint8_t;
struct CDMKeySystemConfiguration;
struct CDMRestrictions;
}

namespace WebKit {

class RemoteCDMInstanceProxy;
struct RemoteCDMInstanceConfiguration;
struct RemoteCDMConfiguration;
struct SharedPreferencesForWebProcess;

class RemoteCDMProxy : public RefCounted<RemoteCDMProxy>, public IPC::MessageReceiver {
public:
    static RefPtr<RemoteCDMProxy> create(RemoteCDMFactoryProxy&, std::unique_ptr<WebCore::CDMPrivate>&&);
    ~RemoteCDMProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    const RemoteCDMConfiguration& configuration() const { return m_configuration.get(); }

    RemoteCDMFactoryProxy* factory() const { return m_factory.get(); }
    RefPtr<RemoteCDMFactoryProxy> protectedFactory() const { return m_factory.get(); }

    bool supportsInitData(const AtomString&, const WebCore::SharedBuffer&);
    RefPtr<WebCore::SharedBuffer> sanitizeResponse(const WebCore::SharedBuffer& response);
    std::optional<String> sanitizeSessionId(const String& sessionId);
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
#endif

private:
    friend class RemoteCDMFactoryProxy;
    RemoteCDMProxy(RemoteCDMFactoryProxy&, std::unique_ptr<WebCore::CDMPrivate>&&, UniqueRef<RemoteCDMConfiguration>&&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    // Messages
    void getSupportedConfiguration(WebCore::CDMKeySystemConfiguration&&, WebCore::CDMPrivate::LocalStorageAccess, CompletionHandler<void(std::optional<WebCore::CDMKeySystemConfiguration>)>&&);
    void createInstance(CompletionHandler<void(std::optional<RemoteCDMInstanceIdentifier>, RemoteCDMInstanceConfiguration&&)>&&);
    void loadAndInitialize();
    void setLogIdentifier(uint64_t);

    WeakPtr<RemoteCDMFactoryProxy> m_factory;
    std::unique_ptr<WebCore::CDMPrivate> m_private;
    UniqueRef<RemoteCDMConfiguration> m_configuration;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
};

}

#endif
