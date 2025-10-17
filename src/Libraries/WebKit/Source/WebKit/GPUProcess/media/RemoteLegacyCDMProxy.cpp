/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
#include "config.h"
#include "RemoteLegacyCDMProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "RemoteLegacyCDMSessionProxy.h"
#include "RemoteMediaPlayerManagerProxy.h"
#include "RemoteMediaPlayerProxy.h"

namespace WebKit {

using namespace WebCore;

Ref<RemoteLegacyCDMProxy> RemoteLegacyCDMProxy::create(WeakPtr<RemoteLegacyCDMFactoryProxy> factory, std::optional<MediaPlayerIdentifier> playerId, Ref<WebCore::LegacyCDM>&& cdm)
{
    return adoptRef(*new RemoteLegacyCDMProxy(WTFMove(factory), playerId, WTFMove(cdm)));
}

RemoteLegacyCDMProxy::RemoteLegacyCDMProxy(WeakPtr<RemoteLegacyCDMFactoryProxy>&& factory, std::optional<MediaPlayerIdentifier> playerId, Ref<WebCore::LegacyCDM>&& cdm)
    : m_factory(WTFMove(factory))
    , m_playerId(playerId)
    , m_cdm(WTFMove(cdm))
{
    m_cdm->setClient(this);
}

RemoteLegacyCDMProxy::~RemoteLegacyCDMProxy()
{
    m_cdm->setClient(nullptr);
}

void RemoteLegacyCDMProxy::supportsMIMEType(const String& mimeType, SupportsMIMETypeCallback&& callback)
{
    callback(protectedCDM()->supportsMIMEType(mimeType));
}

void RemoteLegacyCDMProxy::createSession(const String& keySystem, uint64_t logIdentifier, CreateSessionCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(std::nullopt);
        return;
    }

    auto sessionIdentifier = RemoteLegacyCDMSessionIdentifier::generate();
    Ref session = RemoteLegacyCDMSessionProxy::create(*factory, logIdentifier, sessionIdentifier, protectedCDM());
    factory->addSession(sessionIdentifier, WTFMove(session));
    callback(WTFMove(sessionIdentifier));
}

RefPtr<MediaPlayer> RemoteLegacyCDMProxy::cdmMediaPlayer(const LegacyCDM*) const
{
    RefPtr factory = m_factory.get();
    if (!m_playerId || !factory)
        return nullptr;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return nullptr;

    return gpuConnectionToWebProcess->protectedRemoteMediaPlayerManagerProxy()->mediaPlayer(*m_playerId);
}

std::optional<SharedPreferencesForWebProcess> RemoteLegacyCDMProxy::sharedPreferencesForWebProcess() const
{
    if (!m_factory)
        return std::nullopt;

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_factory->sharedPreferencesForWebProcess();
}

}

#endif
