/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

#include "MessageReceiver.h"
#include "RemoteLegacyCDMFactoryProxy.h"
#include "RemoteLegacyCDMSessionIdentifier.h"
#include <WebCore/LegacyCDMSession.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/UniqueRef.h>

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

namespace WebCore {
class LegacyCDM;
class LegacyCDMSession;
class SharedBuffer;
}

namespace WebKit {

class RemoteLegacyCDMProxy;
class RemoteMediaPlayerProxy;

class RemoteLegacyCDMSessionProxy : public IPC::MessageReceiver, public WebCore::LegacyCDMSessionClient, public RefCounted<RemoteLegacyCDMSessionProxy> {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static Ref<RemoteLegacyCDMSessionProxy> create(RemoteLegacyCDMFactoryProxy&, uint64_t logIdentifier, RemoteLegacyCDMSessionIdentifier, WebCore::LegacyCDM&);
    ~RemoteLegacyCDMSessionProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void invalidate();

    RemoteLegacyCDMFactoryProxy* factory() const { return m_factory.get(); }
    WebCore::LegacyCDMSession* session() const { return m_session.get(); }
    RefPtr<WebCore::LegacyCDMSession> protectedSession() const;

    void setPlayer(WeakPtr<RemoteMediaPlayerProxy>);

    RefPtr<ArrayBuffer> getCachedKeyForKeyId(const String&);
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    friend class RemoteLegacyCDMFactoryProxy;
    RemoteLegacyCDMSessionProxy(RemoteLegacyCDMFactoryProxy&, uint64_t logIdentifier, RemoteLegacyCDMSessionIdentifier, WebCore::LegacyCDM&);

    RefPtr<RemoteLegacyCDMFactoryProxy> protectedFactory() const { return m_factory.get(); }

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    // LegacyCDMSessionClient
    void sendMessage(Uint8Array*, String destinationURL) final;
    void sendError(MediaKeyErrorCode, uint32_t systemCode) final;
    String mediaKeysStorageDirectory() const final;
#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "RemoteLegacyCDMSessionProxy"_s; }
    WTFLogChannel& logChannel() const;
#endif

    // Messages
    using GenerateKeyCallback = CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&, const String&, unsigned short, uint32_t)>;
    void generateKeyRequest(const String& mimeType, RefPtr<WebCore::SharedBuffer>&& initData, GenerateKeyCallback&&);
    void releaseKeys();
    using UpdateCallback = CompletionHandler<void(bool, RefPtr<WebCore::SharedBuffer>&&, unsigned short, uint32_t)>;
    void update(RefPtr<WebCore::SharedBuffer>&& update, UpdateCallback&&);
    using CachedKeyForKeyIDCallback = CompletionHandler<void(RefPtr<WebCore::SharedBuffer>&&)>;
    void cachedKeyForKeyID(String keyId, CachedKeyForKeyIDCallback&&);

    WeakPtr<RemoteLegacyCDMFactoryProxy> m_factory;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif

    RemoteLegacyCDMSessionIdentifier m_identifier;
    RefPtr<WebCore::LegacyCDMSession> m_session;
    WeakPtr<RemoteMediaPlayerProxy> m_player;
};

}

#endif
