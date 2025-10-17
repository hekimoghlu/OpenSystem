/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#import "config.h"
#import "WebCookieCache.h"

#import "NetworkProcessConnection.h"
#import "WebProcess.h"
#import <WebCore/NetworkStorageSession.h>
#import <wtf/text/MakeString.h>

namespace WebKit {

using namespace WebCore;

NetworkStorageSession& WebCookieCache::inMemoryStorageSession()
{
    if (!m_inMemoryStorageSession) {
        String sessionName = makeString("WebKitInProcessStorage-"_s, getCurrentProcessID());
        auto cookieAcceptPolicy = WebProcess::singleton().ensureNetworkProcessConnection().cookieAcceptPolicy();
        auto storageSession = WebCore::createPrivateStorageSession(sessionName.createCFString().get(), cookieAcceptPolicy);
        auto cookieStorage = adoptCF(_CFURLStorageSessionCopyCookieStorage(kCFAllocatorDefault, storageSession.get()));
        m_inMemoryStorageSession = makeUnique<NetworkStorageSession>(WebProcess::singleton().sessionID(), WTFMove(storageSession), WTFMove(cookieStorage), NetworkStorageSession::IsInMemoryCookieStore::Yes);
#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
        m_inMemoryStorageSession->setOptInCookiePartitioningEnabled(m_optInCookiePartitioningEnabled);
#endif
    }
    return *m_inMemoryStorageSession;
}

#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
void WebCookieCache::setOptInCookiePartitioningEnabled(bool enabled)
{
    m_optInCookiePartitioningEnabled = enabled;
    if (m_inMemoryStorageSession)
        m_inMemoryStorageSession->setOptInCookiePartitioningEnabled(enabled);
}
#endif

} // namespace WebKit
