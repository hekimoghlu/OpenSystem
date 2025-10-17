/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include "APIHTTPCookieStore.h"

#include "NetworkProcess.h"
#include "NetworkProcessProxy.h"
#include "SoupCookiePersistentStorageType.h"
#include "WebCookieManagerMessages.h"

namespace API {

void HTTPCookieStore::setCookiePersistentStorage(const WTF::String& storagePath, WebKit::SoupCookiePersistentStorageType storageType)
{
    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->send(Messages::WebCookieManager::SetCookiePersistentStorage(m_sessionID, storagePath, storageType), 0);
}

void HTTPCookieStore::replaceCookies(Vector<WebCore::Cookie>&& cookies, CompletionHandler<void()>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::ReplaceCookies(m_sessionID, cookies), WTFMove(completionHandler));
    else
        completionHandler();
}

void HTTPCookieStore::getAllCookies(CompletionHandler<void(const Vector<WebCore::Cookie>&)>&& completionHandler)
{
    if (auto* networkProcess = networkProcessIfExists())
        networkProcess->sendWithAsyncReply(Messages::WebCookieManager::GetAllCookies(m_sessionID), WTFMove(completionHandler));
    else
        completionHandler({ });
}

} // namespace API
