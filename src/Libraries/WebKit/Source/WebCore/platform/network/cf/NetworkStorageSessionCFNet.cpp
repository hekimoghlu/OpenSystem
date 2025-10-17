/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#include "NetworkStorageSession.h"

#include "PublicSuffixStore.h"
#include "ResourceRequest.h"
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/ProcessID.h>
#include <wtf/ProcessPrivilege.h>

namespace WebCore {

RetainPtr<CFURLStorageSessionRef> NetworkStorageSession::createCFStorageSessionForIdentifier(CFStringRef identifier, ShouldDisableCFURLCache shouldDisableCFURLCache)
{
    auto storageSession = adoptCF(_CFURLStorageSessionCreate(kCFAllocatorDefault, identifier, nullptr));

    if (!storageSession)
        return nullptr;

    if (shouldDisableCFURLCache == ShouldDisableCFURLCache::Yes)
        _CFURLStorageSessionDisableCache(storageSession.get());

    if (shouldDisableCFURLCache == ShouldDisableCFURLCache::No) {
        auto cache = adoptCF(_CFURLStorageSessionCopyCache(kCFAllocatorDefault, storageSession.get()));
        if (!cache)
            return nullptr;

        CFURLCacheSetDiskCapacity(cache.get(), 0);

        auto sharedCache = adoptCF(CFURLCacheCopySharedURLCache());
        CFURLCacheSetMemoryCapacity(cache.get(), CFURLCacheMemoryCapacity(sharedCache.get()));
    }

    if (!NetworkStorageSession::processMayUseCookieAPI())
        return storageSession;

    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));

    auto cookieStorage = adoptCF(_CFURLStorageSessionCopyCookieStorage(kCFAllocatorDefault, storageSession.get()));
    if (!cookieStorage)
        return nullptr;

    auto sharedCookieStorage = _CFHTTPCookieStorageGetDefault(kCFAllocatorDefault);
    auto sharedPolicy = CFHTTPCookieStorageGetCookieAcceptPolicy(sharedCookieStorage);
    CFHTTPCookieStorageSetCookieAcceptPolicy(cookieStorage.get(), sharedPolicy);

    return storageSession;
}

NetworkStorageSession::NetworkStorageSession(PAL::SessionID sessionID, RetainPtr<CFURLStorageSessionRef>&& platformSession, RetainPtr<CFHTTPCookieStorageRef>&& platformCookieStorage, IsInMemoryCookieStore isInMemoryCookieStore)
    : m_sessionID(sessionID)
    , m_platformSession(WTFMove(platformSession))
    , m_isInMemoryCookieStore(isInMemoryCookieStore == IsInMemoryCookieStore::Yes)
{
    ASSERT(processMayUseCookieAPI() || !platformCookieStorage || m_isInMemoryCookieStore);
    m_platformCookieStorage = platformCookieStorage ? WTFMove(platformCookieStorage) : cookieStorage();
}

NetworkStorageSession::NetworkStorageSession(PAL::SessionID sessionID)
    : m_sessionID(sessionID)
{
}

RetainPtr<CFHTTPCookieStorageRef> NetworkStorageSession::cookieStorage() const
{
    if (!processMayUseCookieAPI() && !m_isInMemoryCookieStore)
        return nullptr;

    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies) || m_isInMemoryCookieStore);

    if (m_platformCookieStorage)
        return m_platformCookieStorage;

    if (m_platformSession)
        return adoptCF(_CFURLStorageSessionCopyCookieStorage(kCFAllocatorDefault, m_platformSession.get()));

    // When using NSURLConnection, we also use its shared cookie storage.
    return nullptr;
}

} // namespace WebCore
