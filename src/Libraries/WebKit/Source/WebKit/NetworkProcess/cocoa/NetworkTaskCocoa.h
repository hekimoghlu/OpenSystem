/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

#import "NetworkDataTask.h"
#import <WebCore/FrameIdentifier.h>
#import <WebCore/PageIdentifier.h>
#import <WebCore/ResourceRequest.h>
#import <WebCore/ResourceResponse.h>
#import <WebCore/ShouldRelaxThirdPartyCookieBlocking.h>

OBJC_CLASS NSArray;
OBJC_CLASS NSString;
OBJC_CLASS NSURLSessionTask;

namespace WebCore {
class RegistrableDomain;
enum class ThirdPartyCookieBlockingDecision : uint8_t;
}

namespace WebKit {

class NetworkTaskCocoa {
public:
    virtual ~NetworkTaskCocoa() = default;

    void willPerformHTTPRedirection(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&, RedirectCompletionHandler&&);
    virtual std::optional<WebCore::FrameIdentifier> frameID() const = 0;
    virtual std::optional<WebCore::PageIdentifier> pageID() const = 0;
    virtual std::optional<WebPageProxyIdentifier> webPageProxyID() const = 0;

    WebCore::ShouldRelaxThirdPartyCookieBlocking shouldRelaxThirdPartyCookieBlocking() const;

protected:
    NetworkTaskCocoa(NetworkSession&);

    static NSHTTPCookieStorage *statelessCookieStorage();
    bool shouldApplyCookiePolicyForThirdPartyCloaking() const;
    enum class IsRedirect : bool { No, Yes };
    void setCookieTransform(const WebCore::ResourceRequest&, IsRedirect);
    void blockCookies();
    void unblockCookies();
    static void updateTaskWithFirstPartyForSameSiteCookies(NSURLSessionTask*, const WebCore::ResourceRequest&);
#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
    void updateTaskWithStoragePartitionIdentifier(const WebCore::ResourceRequest&);
#endif
    bool needsFirstPartyCookieBlockingLatchModeQuirk(const URL& firstPartyURL, const URL& requestURL, const URL& redirectingURL) const;
    static NSString *lastRemoteIPAddress(NSURLSessionTask *);
    static WebCore::RegistrableDomain lastCNAMEDomain(String);
    static bool shouldBlockCookies(WebCore::ThirdPartyCookieBlockingDecision);
    WebCore::ThirdPartyCookieBlockingDecision requestThirdPartyCookieBlockingDecision(const WebCore::ResourceRequest&) const;
#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
    bool isOptInCookiePartitioningEnabled() const;
#endif

    bool isAlwaysOnLoggingAllowed() const { return m_isAlwaysOnLoggingAllowed; }
    virtual NSURLSessionTask* task() const = 0;
    virtual WebCore::StoredCredentialsPolicy storedCredentialsPolicy() const = 0;

private:
    void setCookieTransformForFirstPartyRequest(const WebCore::ResourceRequest&);
    void setCookieTransformForThirdPartyRequest(const WebCore::ResourceRequest&, IsRedirect);

    WeakPtr<NetworkSession> m_networkSession;
    bool m_hasBeenSetToUseStatelessCookieStorage { false };
    Seconds m_ageCapForCNAMECloakedCookies { 24_h * 7 };
    bool m_isAlwaysOnLoggingAllowed { false };
};

} // namespace WebKit
