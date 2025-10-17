/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
#include "SecurityPolicy.h"

#include "OriginAccessEntry.h"
#include "OriginAccessPatterns.h"
#include "SecurityOrigin.h"
#include "UserContentURLPattern.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

static SecurityPolicy::LocalLoadPolicy localLoadPolicy = SecurityPolicy::AllowLocalLoadsForLocalOnly;

using OriginAccessAllowlist = Vector<OriginAccessEntry>;
using OriginAccessMap = UncheckedKeyHashMap<SecurityOriginData, std::unique_ptr<OriginAccessAllowlist>>;

static Lock originAccessMapLock;
static OriginAccessMap& originAccessMap() WTF_REQUIRES_LOCK(originAccessMapLock)
{
    ASSERT(originAccessMapLock.isHeld());
    static NeverDestroyed<OriginAccessMap> originAccessMap;
    return originAccessMap;
}

bool SecurityPolicy::shouldHideReferrer(const URL& url, const URL& referrer)
{
    bool referrerIsSecureURL = referrer.protocolIs("https"_s);
    bool referrerIsWebURL = referrerIsSecureURL || referrer.protocolIs("http"_s);

    if (!referrerIsWebURL)
        return true;

    if (!referrerIsSecureURL)
        return false;

    bool URLIsSecureURL = url.protocolIs("https"_s);

    return !URLIsSecureURL;
}

String SecurityPolicy::referrerToOriginString(const URL& referrer)
{
    String originString = SecurityOrigin::create(referrer)->toString();
    if (originString == "null"_s)
        return String();
    // A security origin is not a canonical URL as it lacks a path. Add /
    // to turn it into a canonical URL we can use as referrer.
    return makeString(originString, '/');
}

String SecurityPolicy::generateReferrerHeader(ReferrerPolicy referrerPolicy, const URL& url, const URL& referrer, const OriginAccessPatterns& patterns)
{
    ASSERT(referrer.string() == URL { referrer.string() }.strippedForUseAsReferrer().string
        || referrer.string() == SecurityOrigin::create(URL { referrer.string() })->toString());

    if (referrer.isEmpty())
        return String();

    if (!referrer.protocolIsInHTTPFamily())
        return String();

    switch (referrerPolicy) {
    case ReferrerPolicy::EmptyString:
        ASSERT_NOT_REACHED();
        break;
    case ReferrerPolicy::NoReferrer:
        return String();
    case ReferrerPolicy::NoReferrerWhenDowngrade:
        break;
    case ReferrerPolicy::SameOrigin: {
        auto origin = SecurityOrigin::create(referrer);
        if (!origin->canRequest(url, patterns))
            return String();
        break;
    }
    case ReferrerPolicy::Origin:
        return referrerToOriginString(referrer);
    case ReferrerPolicy::StrictOrigin:
        if (shouldHideReferrer(url, referrer))
            return String();
        return referrerToOriginString(referrer);
    case ReferrerPolicy::OriginWhenCrossOrigin: {
        auto origin = SecurityOrigin::create(referrer);
        if (!origin->canRequest(url, patterns))
            return referrerToOriginString(referrer);
        break;
    }
    case ReferrerPolicy::StrictOriginWhenCrossOrigin: {
        auto origin = SecurityOrigin::create(referrer);
        if (!origin->canRequest(url, patterns)) {
            if (shouldHideReferrer(url, referrer))
                return String();
            return referrerToOriginString(referrer);
        }
        break;
    }
    case ReferrerPolicy::UnsafeUrl:
        return referrer.string();
    }

    return shouldHideReferrer(url, referrer) ? String() : referrer.string();
}

String SecurityPolicy::generateOriginHeader(ReferrerPolicy referrerPolicy, const URL& url, const SecurityOrigin& securityOrigin, const OriginAccessPatterns& patterns)
{
    switch (referrerPolicy) {
    case ReferrerPolicy::NoReferrer:
        return "null"_s;
    case ReferrerPolicy::NoReferrerWhenDowngrade:
    case ReferrerPolicy::StrictOrigin:
    case ReferrerPolicy::StrictOriginWhenCrossOrigin:
        if (protocolIs(securityOrigin.protocol(), "https"_s) && !url.protocolIs("https"_s))
            return "null"_s;
        break;
    case ReferrerPolicy::SameOrigin:
        if (!securityOrigin.canRequest(url, patterns))
            return "null"_s;
        break;
    case ReferrerPolicy::EmptyString:
    case ReferrerPolicy::Origin:
    case ReferrerPolicy::OriginWhenCrossOrigin:
    case ReferrerPolicy::UnsafeUrl:
        break;
    }

    return securityOrigin.toString();
}

bool SecurityPolicy::shouldInheritSecurityOriginFromOwner(const URL& url)
{
    // Paraphrased from <https://html.spec.whatwg.org/multipage/browsers.html#origin> (8 July 2016)
    //
    // If a Document has the address "about:blank"
    //      The origin of the document is the origin it was assigned when its browsing context was created.
    // If a Document has the address "about:srcdoc"
    //      The origin of the document is the origin of its parent document.
    //
    // Note: We generalize this to invalid URLs because we treat such URLs as about:blank.
    // FIXME: We also allow some URLs like "about:BLANK". We should probably block navigation to these URLs, see rdar://problem/57966056.
    return url.isEmpty() || url.isAboutBlank() || url.isAboutSrcDoc() || equalIgnoringASCIICase(url.string(), aboutBlankURL().string());
}

bool SecurityPolicy::isBaseURLSchemeAllowed(const URL& url)
{
    // See <https://github.com/whatwg/html/issues/2249>.
    return !url.protocolIsData() && !url.protocolIsJavaScript();
}

void SecurityPolicy::setLocalLoadPolicy(LocalLoadPolicy policy)
{
    localLoadPolicy = policy;
}

bool SecurityPolicy::restrictAccessToLocal()
{
    return localLoadPolicy != SecurityPolicy::AllowLocalLoadsForAll;
}

bool SecurityPolicy::allowSubstituteDataAccessToLocal()
{
    return localLoadPolicy != SecurityPolicy::AllowLocalLoadsForLocalOnly;
}

bool SecurityPolicy::isAccessAllowed(const SecurityOrigin& activeOrigin, const SecurityOrigin& targetOrigin, const URL& targetURL, const OriginAccessPatterns& patterns)
{
    ASSERT(targetOrigin.equal(SecurityOrigin::create(targetURL)));
    {
        Locker locker { originAccessMapLock };
        if (auto* list = originAccessMap().get(activeOrigin.data())) {
            for (auto& entry : *list) {
                if (entry.matchesOrigin(targetOrigin))
                    return true;
            }
        }
    }

    return patterns.anyPatternMatches(targetURL);
}

bool SecurityPolicy::isAccessAllowed(const SecurityOrigin& activeOrigin, const URL& url, const OriginAccessPatterns& patterns)
{
    return isAccessAllowed(activeOrigin, SecurityOrigin::create(url).get(), url, patterns);
}

void SecurityPolicy::addOriginAccessAllowlistEntry(const SecurityOrigin& sourceOrigin, const String& destinationProtocol, const String& destinationDomain, bool allowDestinationSubdomains)
{
    ASSERT(!sourceOrigin.isOpaque());
    if (sourceOrigin.isOpaque())
        return;

    Locker locker { originAccessMapLock };
    auto* allowList = originAccessMap().ensure(sourceOrigin.data(), [] { return makeUnique<OriginAccessAllowlist>(); }).iterator->value.get();
    allowList->append(OriginAccessEntry(destinationProtocol, destinationDomain, allowDestinationSubdomains ? OriginAccessEntry::AllowSubdomains : OriginAccessEntry::DisallowSubdomains, OriginAccessEntry::TreatIPAddressAsIPAddress));
}

void SecurityPolicy::removeOriginAccessAllowlistEntry(const SecurityOrigin& sourceOrigin, const String& destinationProtocol, const String& destinationDomain, bool allowDestinationSubdomains)
{
    ASSERT(!sourceOrigin.isOpaque());
    if (sourceOrigin.isOpaque())
        return;

    Locker locker { originAccessMapLock };
    auto& map = originAccessMap();
    auto it = map.find(sourceOrigin.data());
    if (it == map.end())
        return;

    auto& list = *it->value;
    OriginAccessEntry originAccessEntry(destinationProtocol, destinationDomain, allowDestinationSubdomains ? OriginAccessEntry::AllowSubdomains : OriginAccessEntry::DisallowSubdomains, OriginAccessEntry::TreatIPAddressAsIPAddress);
    if (!list.removeFirst(originAccessEntry))
        return;

    if (list.isEmpty())
        map.remove(it);
}

void SecurityPolicy::resetOriginAccessAllowlists()
{
    Locker locker { originAccessMapLock };
    originAccessMap().clear();
}

} // namespace WebCore
