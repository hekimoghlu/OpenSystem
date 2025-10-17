/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

#include "WebPageProxyIdentifier.h"
#include <WebCore/CookieChangeListener.h>
#include <WebCore/CookieJar.h>
#include <WebCore/SameSiteInfo.h>
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/RefCounter.h>

namespace WebCore {
struct Cookie;
class NetworkStorageSession;
enum class ShouldRelaxThirdPartyCookieBlocking : bool;
}

namespace WebKit {

enum PendingCookieUpdateCounterType { };
using PendingCookieUpdateCounter = RefCounter<PendingCookieUpdateCounterType>;

class WebCookieCache final : public WebCore::CookieChangeListener {
public:
    WebCookieCache() = default;
    virtual ~WebCookieCache();

    bool isSupported();

    String cookiesForDOM(const URL& firstParty, const WebCore::SameSiteInfo&, const URL&, WebCore::FrameIdentifier, WebCore::PageIdentifier, WebPageProxyIdentifier, WebCore::IncludeSecureCookies);
    void setCookiesFromDOM(const URL& firstParty, const WebCore::SameSiteInfo&, const URL&, WebCore::FrameIdentifier, WebCore::PageIdentifier, const String& cookieString, WebCore::ShouldRelaxThirdPartyCookieBlocking);

    PendingCookieUpdateCounter::Token WARN_UNUSED_RETURN willSetCookieFromDOM();
    void didSetCookieFromDOM(PendingCookieUpdateCounter::Token, const URL& firstParty, const WebCore::SameSiteInfo&, const URL&, WebCore::FrameIdentifier, WebCore::PageIdentifier, const WebCore::Cookie&, WebCore::ShouldRelaxThirdPartyCookieBlocking);

    void allCookiesDeleted();

    void clear();
    void clearForHost(const String&);

    void setOptInCookiePartitioningEnabled(bool);

private:
    WebCore::NetworkStorageSession& inMemoryStorageSession();
    void pruneCacheIfNecessary();
    bool cacheMayBeOutOfSync() const;

    // CookieChangeListener
    void cookiesAdded(const String& host, const Vector<WebCore::Cookie>&) final;
    void cookiesDeleted(const String& host, const Vector<WebCore::Cookie>&) final;

    HashSet<String> m_hostsWithInMemoryStorage;
    std::unique_ptr<WebCore::NetworkStorageSession> m_inMemoryStorageSession;
#if HAVE(ALLOW_ONLY_PARTITIONED_COOKIES)
    bool m_optInCookiePartitioningEnabled { false };
#endif

    PendingCookieUpdateCounter m_pendingCookieUpdateCounter;
};

} // namespace WebKit
