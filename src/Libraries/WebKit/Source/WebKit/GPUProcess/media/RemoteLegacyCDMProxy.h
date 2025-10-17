/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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

#include "MessageReceiver.h"
#include "RemoteLegacyCDMFactoryProxy.h"
#include "RemoteLegacyCDMIdentifier.h"
#include <WebCore/LegacyCDM.h>
#include <wtf/Forward.h>
#include <wtf/Markable.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class RemoteLegacyCDMProxy : public IPC::MessageReceiver, public WebCore::LegacyCDMClient, public RefCounted<RemoteLegacyCDMProxy> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteLegacyCDMProxy);
public:
    static Ref<RemoteLegacyCDMProxy> create(WeakPtr<RemoteLegacyCDMFactoryProxy>, std::optional<WebCore::MediaPlayerIdentifier>, Ref<WebCore::LegacyCDM>&&);
    ~RemoteLegacyCDMProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    RemoteLegacyCDMFactoryProxy* factory() const { return m_factory.get(); }
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    friend class RemoteLegacyCDMFactoryProxy;
    RemoteLegacyCDMProxy(WeakPtr<RemoteLegacyCDMFactoryProxy>&&, std::optional<WebCore::MediaPlayerIdentifier>, Ref<WebCore::LegacyCDM>&&);

    RefPtr<RemoteLegacyCDMFactoryProxy> protectedFactory() const { return m_factory.get(); }

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    // Messages
    using SupportsMIMETypeCallback = CompletionHandler<void(bool)>;
    void supportsMIMEType(const String&, SupportsMIMETypeCallback&&);
    using CreateSessionCallback = CompletionHandler<void(std::optional<RemoteLegacyCDMSessionIdentifier>&&)>;
    void createSession(const String&, uint64_t, CreateSessionCallback&&);
    void setPlayerId(std::optional<WebCore::MediaPlayerIdentifier> playerId) { m_playerId = playerId; }

    // LegacyCDMClient
    RefPtr<WebCore::MediaPlayer> cdmMediaPlayer(const WebCore::LegacyCDM*) const final;

    Ref<WebCore::LegacyCDM> protectedCDM() const { return m_cdm; }

    WeakPtr<RemoteLegacyCDMFactoryProxy> m_factory;
    Markable<WebCore::MediaPlayerIdentifier> m_playerId;
    Ref<WebCore::LegacyCDM> m_cdm;
};

}

#endif
