/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#include "RemoteMediaPlayerMIMETypeCache.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "Logging.h"
#include "RemoteMediaPlayerManager.h"
#include "RemoteMediaPlayerManagerProxyMessages.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaPlayerMIMETypeCache);

RemoteMediaPlayerMIMETypeCache::RemoteMediaPlayerMIMETypeCache(RemoteMediaPlayerManager& manager, MediaPlayerEnums::MediaEngineIdentifier engineIdentifier)
    : m_manager(manager)
    , m_engineIdentifier(engineIdentifier)
{
}

void RemoteMediaPlayerMIMETypeCache::addSupportedTypes(const Vector<String>& newTypes)
{
    m_supportedTypesCache.add(newTypes.begin(), newTypes.end());
}

bool RemoteMediaPlayerMIMETypeCache::isEmpty() const
{
    return m_hasPopulatedSupportedTypesCacheFromGPUProcess && m_supportedTypesCache.isEmpty();
}

HashSet<String>& RemoteMediaPlayerMIMETypeCache::supportedTypes()
{
    ASSERT(isMainRunLoop());
    if (!m_hasPopulatedSupportedTypesCacheFromGPUProcess) {
        auto sendResult = protectedManager()->gpuProcessConnection().connection().sendSync(Messages::RemoteMediaPlayerManagerProxy::GetSupportedTypes(m_engineIdentifier), 0);
        if (sendResult.succeeded()) {
            auto& [types] = sendResult.reply();
            addSupportedTypes(types);
            m_hasPopulatedSupportedTypesCacheFromGPUProcess = true;
        } else
            RELEASE_LOG_ERROR(Media, "RemoteMediaPlayerMIMETypeCache::supportedTypes: Sync IPC to the GPUProcess failed with error %" PUBLIC_LOG_STRING, IPC::errorAsString(sendResult.error()).characters());
    }
    return m_supportedTypesCache;
}

MediaPlayerEnums::SupportsType RemoteMediaPlayerMIMETypeCache::supportsTypeAndCodecs(const MediaEngineSupportParameters& parameters)
{
    if (parameters.type.raw().isEmpty())
        return MediaPlayerEnums::SupportsType::MayBeSupported;

    SupportedTypesAndCodecsKey searchKey { parameters.type.raw(), parameters.isMediaSource, parameters.isMediaStream, parameters.requiresRemotePlayback };

    if (m_supportsTypeAndCodecsCache) {
        auto it = m_supportsTypeAndCodecsCache->find(searchKey);
        if (it != m_supportsTypeAndCodecsCache->end())
            return it->value;
    }

    if (!m_supportsTypeAndCodecsCache)
        m_supportsTypeAndCodecsCache = HashMap<SupportedTypesAndCodecsKey, MediaPlayerEnums::SupportsType> { };

    auto sendResult = protectedManager()->gpuProcessConnection().connection().sendSync(Messages::RemoteMediaPlayerManagerProxy::SupportsTypeAndCodecs(m_engineIdentifier, parameters), 0);
    auto [result] = sendResult.takeReplyOr(MediaPlayerEnums::SupportsType::IsNotSupported);
    if (sendResult.succeeded())
        m_supportsTypeAndCodecsCache->add(searchKey, result);

    return result;
}

Ref<RemoteMediaPlayerManager> RemoteMediaPlayerMIMETypeCache::protectedManager() const
{
    return m_manager.get().releaseNonNull();
}

}

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
