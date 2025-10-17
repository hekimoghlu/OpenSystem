/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#import "WebCookieManager.h"

#import "NetworkProcess.h"
#import "NetworkSession.h"
#import <WebCore/HTTPCookieAcceptPolicy.h>
#import <WebCore/NetworkStorageSession.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/CallbackAggregator.h>
#import <wtf/ProcessPrivilege.h>

namespace WebKit {
using namespace WebCore;

static CFHTTPCookieStorageAcceptPolicy toCFHTTPCookieStorageAcceptPolicy(HTTPCookieAcceptPolicy policy)
{
    switch (policy) {
    case HTTPCookieAcceptPolicy::AlwaysAccept:
        return CFHTTPCookieStorageAcceptPolicyAlways;
    case HTTPCookieAcceptPolicy::Never:
        return CFHTTPCookieStorageAcceptPolicyNever;
    case HTTPCookieAcceptPolicy::OnlyFromMainDocumentDomain:
        return CFHTTPCookieStorageAcceptPolicyOnlyFromMainDocumentDomain;
    case HTTPCookieAcceptPolicy::ExclusivelyFromMainDocumentDomain:
        return CFHTTPCookieStorageAcceptPolicyExclusivelyFromMainDocumentDomain;
    }
    ASSERT_NOT_REACHED();

    return CFHTTPCookieStorageAcceptPolicyAlways;
}

void WebCookieManager::platformSetHTTPCookieAcceptPolicy(PAL::SessionID sessionID, HTTPCookieAcceptPolicy policy, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(hasProcessPrivilege(ProcessPrivilege::CanAccessRawCookies));

    auto* storageSession = protectedProcess()->storageSession(sessionID);
    if (!storageSession)
        return completionHandler();

    RetainPtr nsCookieStorage = storageSession->nsCookieStorage();
    if (!nsCookieStorage)
        return completionHandler();

    CFHTTPCookieStorageSetCookieAcceptPolicy([nsCookieStorage _cookieStorage], toCFHTTPCookieStorageAcceptPolicy(policy));
    saveCookies(nsCookieStorage.get(), WTFMove(completionHandler));
}

} // namespace WebKit
