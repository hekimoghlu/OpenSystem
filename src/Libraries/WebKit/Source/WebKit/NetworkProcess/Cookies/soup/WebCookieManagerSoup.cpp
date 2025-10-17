/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 22, 2025.
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
#include "WebCookieManager.h"

#include "NetworkProcess.h"
#include "NetworkSessionSoup.h"
#include "SoupCookiePersistentStorageType.h"
#include <WebCore/HTTPCookieAcceptPolicy.h>
#include <WebCore/NetworkStorageSession.h>
#include <WebCore/SoupNetworkSession.h>
#include <libsoup/soup.h>

namespace WebKit {
using namespace WebCore;

void WebCookieManager::platformSetHTTPCookieAcceptPolicy(PAL::SessionID sessionID, HTTPCookieAcceptPolicy policy, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->setCookieAcceptPolicy(policy);

    completionHandler();
}

void WebCookieManager::setCookiePersistentStorage(PAL::SessionID sessionID, const String& storagePath, SoupCookiePersistentStorageType storageType)
{
    if (auto* networkSession = protectedProcess()->networkSession(sessionID))
        static_cast<NetworkSessionSoup&>(*networkSession).setCookiePersistentStorage(storagePath, storageType);
}

void WebCookieManager::replaceCookies(PAL::SessionID sessionID, const Vector<WebCore::Cookie>& cookies, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->replaceCookies(cookies);
    completionHandler();
}

} // namespace WebKit
