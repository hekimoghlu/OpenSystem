/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
#include "RemoteLegacyCDM.h"

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "RemoteLegacyCDMFactory.h"
#include "RemoteLegacyCDMProxyMessages.h"
#include "RemoteLegacyCDMSession.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteLegacyCDM);

RemoteLegacyCDM::RemoteLegacyCDM(RemoteLegacyCDMFactory& factory, RemoteLegacyCDMIdentifier identifier)
    : m_factory(factory)
    , m_identifier(identifier)
{
}

RemoteLegacyCDM::~RemoteLegacyCDM() = default;

Ref<RemoteLegacyCDMFactory> RemoteLegacyCDM::protectedFactory() const
{
    return m_factory.get();
}

bool RemoteLegacyCDM::supportsMIMEType(const String& mimeType) const
{
    auto sendResult = protectedFactory()->gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMProxy::SupportsMIMEType(mimeType), m_identifier);
    auto [supported] = sendResult.takeReplyOr(false);
    return supported;
}

RefPtr<WebCore::LegacyCDMSession> RemoteLegacyCDM::createSession(WebCore::LegacyCDMSessionClient& client)
{
    String storageDirectory = client.mediaKeysStorageDirectory();

    uint64_t logIdentifier { 0 };
#if !RELEASE_LOG_DISABLED
    logIdentifier = reinterpret_cast<uint64_t>(client.logIdentifier());
#endif

    Ref factory = m_factory.get();
    auto sendResult = factory->gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMProxy::CreateSession(storageDirectory, logIdentifier), m_identifier);
    auto [identifier] = sendResult.takeReplyOr(std::nullopt);
    if (!identifier)
        return nullptr;
    return RemoteLegacyCDMSession::create(factory, WTFMove(*identifier), client);
}

void RemoteLegacyCDM::setPlayerId(std::optional<MediaPlayerIdentifier> identifier)
{
    protectedFactory()->gpuProcessConnection().connection().send(Messages::RemoteLegacyCDMProxy::SetPlayerId(identifier), m_identifier);
}

void RemoteLegacyCDM::ref() const
{
    m_factory->ref();
}

void RemoteLegacyCDM::deref() const
{
    m_factory->deref();
}

}

#endif
