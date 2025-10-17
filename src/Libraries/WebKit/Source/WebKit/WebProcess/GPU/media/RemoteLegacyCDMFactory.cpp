/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "RemoteLegacyCDMFactory.h"

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "MediaPlayerPrivateRemote.h"
#include "RemoteLegacyCDM.h"
#include "RemoteLegacyCDMFactoryProxyMessages.h"
#include "RemoteLegacyCDMSession.h"
#include "RemoteLegacyCDMSessionMessages.h"
#include "WebProcess.h"
#include <WebCore/LegacyCDM.h>
#include <WebCore/Settings.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteLegacyCDMFactory);

RemoteLegacyCDMFactory::RemoteLegacyCDMFactory(WebProcess& webProcess)
    : m_webProcess(webProcess)
{
}

RemoteLegacyCDMFactory::~RemoteLegacyCDMFactory() = default;

void RemoteLegacyCDMFactory::ref() const
{
    m_webProcess->ref();
}

void RemoteLegacyCDMFactory::deref() const
{
    m_webProcess->deref();
}

void RemoteLegacyCDMFactory::registerFactory()
{
    LegacyCDM::clearFactories();
    LegacyCDM::registerCDMFactory(
        [protectedThis = Ref { *this }] (LegacyCDM& privateCDM) -> std::unique_ptr<WebCore::CDMPrivateInterface> {
            return protectedThis->createCDM(privateCDM);
        },
        [protectedThis = Ref { *this }] (const String& keySystem) {
            return protectedThis->supportsKeySystem(keySystem);
        },
        [protectedThis = Ref { *this }] (const String& keySystem, const String& mimeType) {
            return protectedThis->supportsKeySystemAndMimeType(keySystem, mimeType);
        }
    );
}

ASCIILiteral RemoteLegacyCDMFactory::supplementName()
{
    return "RemoteLegacyCDMFactory"_s;
}

GPUProcessConnection& RemoteLegacyCDMFactory::gpuProcessConnection()
{
    return WebProcess::singleton().ensureGPUProcessConnection();
}

bool RemoteLegacyCDMFactory::supportsKeySystem(const String& keySystem)
{
    auto foundInCache = m_supportsKeySystemCache.find(keySystem);
    if (foundInCache != m_supportsKeySystemCache.end())
        return foundInCache->value;

    auto sendResult = gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMFactoryProxy::SupportsKeySystem(keySystem, std::nullopt), { });
    auto [supported] = sendResult.takeReplyOr(false);
    m_supportsKeySystemCache.set(keySystem, supported);
    return supported;
}

bool RemoteLegacyCDMFactory::supportsKeySystemAndMimeType(const String& keySystem, const String& mimeType)
{
    auto key = std::make_pair(keySystem, mimeType);
    auto foundInCache = m_supportsKeySystemAndMimeTypeCache.find(key);
    if (foundInCache != m_supportsKeySystemAndMimeTypeCache.end())
        return foundInCache->value;

    auto sendResult = gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMFactoryProxy::SupportsKeySystem(keySystem, mimeType), { });
    auto [supported] = sendResult.takeReplyOr(false);
    m_supportsKeySystemAndMimeTypeCache.set(key, supported);
    return supported;
}

std::unique_ptr<CDMPrivateInterface> RemoteLegacyCDMFactory::createCDM(WebCore::LegacyCDM& cdm)
{
    std::optional<MediaPlayerIdentifier> playerId;
    if (auto player = cdm.mediaPlayer())
        playerId = gpuProcessConnection().mediaPlayerManager().findRemotePlayerId(player->playerPrivate());

    auto sendResult = gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMFactoryProxy::CreateCDM(cdm.keySystem(), WTFMove(playerId)), { });
    auto [identifier] = sendResult.takeReplyOr(std::nullopt);
    if (!identifier)
        return nullptr;
    auto remoteCDM = makeUniqueRefWithoutRefCountedCheck<RemoteLegacyCDM>(*this, *identifier);
    m_cdms.set(*identifier, remoteCDM.get());
    return remoteCDM.moveToUniquePtr();
}

void RemoteLegacyCDMFactory::addSession(RemoteLegacyCDMSessionIdentifier identifier, RemoteLegacyCDMSession& session)
{
    ASSERT(!m_sessions.contains(identifier));
    m_sessions.set(identifier, WeakPtr { session });

    gpuProcessConnection().messageReceiverMap().addMessageReceiver(Messages::RemoteLegacyCDMSession::messageReceiverName(), identifier.toUInt64(), session);
}

void RemoteLegacyCDMFactory::removeSession(RemoteLegacyCDMSessionIdentifier identifier)
{
    ASSERT(m_sessions.contains(identifier));
    RefPtr session = m_sessions.get(identifier).get();
    gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteLegacyCDMFactoryProxy::RemoveSession(identifier), [protectedThis = Ref { *this }, identifier, session = WTFMove(session)] {
        ASSERT(protectedThis->m_sessions.contains(identifier));
        protectedThis->m_sessions.remove(identifier);
        protectedThis->gpuProcessConnection().messageReceiverMap().removeMessageReceiver(Messages::RemoteLegacyCDMSession::messageReceiverName(), identifier.toUInt64());
        UNUSED_PARAM(session);
    }, { });
}

RemoteLegacyCDM* RemoteLegacyCDMFactory::findCDM(CDMPrivateInterface* privateInterface) const
{
    for (auto& cdm : m_cdms.values()) {
        if (privateInterface == cdm.get())
            return cdm.get();
    }
    return nullptr;
}

}

#endif
