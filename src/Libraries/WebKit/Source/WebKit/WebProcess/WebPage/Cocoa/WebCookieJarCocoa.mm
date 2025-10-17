/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#import "WebCookieJar.h"

#if PLATFORM(COCOA)

#import <WebCore/Document.h>
#import <WebCore/DocumentInlines.h>
#import <WebCore/Quirks.h>
#import <WebCore/SameSiteInfo.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <wtf/text/StringBuilder.h>
#import <wtf/text/cf/StringConcatenateCF.h>

namespace WebKit {

static RetainPtr<NSDictionary> policyProperties(const WebCore::SameSiteInfo& sameSiteInfo, const URL& url)
{
    static NSURL *emptyURL = [[NSURL alloc] initWithString:@""];
    NSURL *nsURL = url;
    NSDictionary *policyProperties = @{
        @"_kCFHTTPCookiePolicyPropertySiteForCookies": sameSiteInfo.isSameSite ? nsURL : emptyURL,
        @"_kCFHTTPCookiePolicyPropertyIsTopLevelNavigation": [NSNumber numberWithBool:sameSiteInfo.isTopSite],
    };
    return policyProperties;
}

String WebCookieJar::cookiesInPartitionedCookieStorage(const WebCore::Document& document, const URL& cookieURL, const WebCore::SameSiteInfo& sameSiteInfo) const
{
    if (!document.quirks().shouldUseEphemeralPartitionedStorageForDOMCookies(cookieURL))
        return { };

    if (!m_partitionedStorageForDOMCookies)
        return { };

    URL firstPartyURL = document.firstPartyForCookies();
    String partition = WebCore::RegistrableDomain(firstPartyURL).string();
    if (partition.isEmpty())
        return { };

    __block RetainPtr<NSArray> cookies;
    [m_partitionedStorageForDOMCookies.get() _getCookiesForURL:cookieURL mainDocumentURL:firstPartyURL partition:partition policyProperties:policyProperties(sameSiteInfo, cookieURL).get() completionHandler:^(NSArray *result) {
        cookies = result;
    }];

    if (!cookies || ![cookies.get() count])
        return { };

    StringBuilder cookiesBuilder;
    for (NSHTTPCookie *cookie in cookies.get()) {
        if (![[cookie name] length] || [cookie isHTTPOnly])
            continue;

        cookiesBuilder.append(cookiesBuilder.isEmpty() ? ""_s : "; "_s, [cookie name], '=', [cookie value]);
    }

    return cookiesBuilder.toString();
}

void WebCookieJar::setCookiesInPartitionedCookieStorage(const WebCore::Document& document, const URL& cookieURL, const WebCore::SameSiteInfo& sameSiteInfo, const String& cookieString)
{
    if (!document.quirks().shouldUseEphemeralPartitionedStorageForDOMCookies(cookieURL))
        return;

    if (cookieString.isEmpty())
        return;

    auto firstPartyURL = document.firstPartyForCookies();
    String partition = WebCore::RegistrableDomain(firstPartyURL).string();
    if (partition.isEmpty())
        return;

    NSHTTPCookie *cookie = [NSHTTPCookie _cookieForSetCookieString:cookieString forURL:cookieURL partition:partition];
    if (!cookie || ![[cookie name] length] || [cookie isHTTPOnly])
        return;

    [ensurePartitionedCookieStorage() _setCookies:@[cookie] forURL:cookieURL mainDocumentURL:firstPartyURL policyProperties:policyProperties(sameSiteInfo, cookieURL).get()];
}

NSHTTPCookieStorage* WebCookieJar::ensurePartitionedCookieStorage()
{
    if (!m_partitionedStorageForDOMCookies) {
        m_partitionedStorageForDOMCookies = adoptNS([[NSHTTPCookieStorage alloc] _initWithIdentifier:@"WebCookieJar" private:true]);
        m_partitionedStorageForDOMCookies.get().cookieAcceptPolicy = NSHTTPCookieAcceptPolicyAlways;
    }

    return m_partitionedStorageForDOMCookies.get();
}

} // namespace WebKit
#endif
