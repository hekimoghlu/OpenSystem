/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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

#include "AuthenticationChallengeDisposition.h"
#include "AuthenticationChallengeProxy.h"
#include "AuthenticationDecisionListener.h"
#include "DownloadID.h"
#include "DownloadProxy.h"
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class ResourceError;
class ResourceRequest;
class ResourceResponse;
}

namespace WebKit {
class AuthenticationChallengeProxy;
class WebsiteDataStore;
class WebProtectionSpace;

enum class AllowOverwrite : bool;
}

namespace API {

class Data;

class DownloadClient : public RefCounted<DownloadClient> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DownloadClient);
public:
    virtual ~DownloadClient() { }

    virtual void legacyDidStart(WebKit::DownloadProxy&) { }
    virtual void didReceiveAuthenticationChallenge(WebKit::DownloadProxy&, WebKit::AuthenticationChallengeProxy& challenge) { challenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::Cancel); }
    virtual void didReceiveData(WebKit::DownloadProxy&, uint64_t, uint64_t, uint64_t) { }
    virtual void decidePlaceholderPolicy(WebKit::DownloadProxy&, CompletionHandler<void(WebKit::UseDownloadPlaceholder, const WTF::URL&)>&& completionHandler) { completionHandler(WebKit::UseDownloadPlaceholder::No, { }); }
    virtual void decideDestinationWithSuggestedFilename(WebKit::DownloadProxy&, const WebCore::ResourceResponse&, const WTF::String&, CompletionHandler<void(WebKit::AllowOverwrite, WTF::String)>&& completionHandler) { completionHandler(WebKit::AllowOverwrite::No, { }); }
    virtual void didCreateDestination(WebKit::DownloadProxy&, const WTF::String&) { }
    virtual void didFinish(WebKit::DownloadProxy&) { }
    virtual void didFail(WebKit::DownloadProxy&, const WebCore::ResourceError&, API::Data* resumeData) { }
#if HAVE(MODERN_DOWNLOADPROGRESS)
    virtual void didReceivePlaceholderURL(WebKit::DownloadProxy&, const WTF::URL&, std::span<const uint8_t>, CompletionHandler<void()>&& completionHandler) { completionHandler(); }
    virtual void didReceiveFinalURL(WebKit::DownloadProxy&, const WTF::URL&, std::span<const uint8_t>) { }
#endif
    virtual void legacyDidCancel(WebKit::DownloadProxy&) { }
    virtual void processDidCrash(WebKit::DownloadProxy&) { }
    virtual void willSendRequest(WebKit::DownloadProxy&, WebCore::ResourceRequest&& request, const WebCore::ResourceResponse&, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler) { completionHandler(WTFMove(request)); }
};

} // namespace API
