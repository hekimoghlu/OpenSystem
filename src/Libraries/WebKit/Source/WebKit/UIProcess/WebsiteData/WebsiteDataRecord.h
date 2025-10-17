/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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

#include "WebsiteDataType.h"
#include <WebCore/RegistrableDomain.h>
#include <WebCore/SecurityOriginData.h>
#include <WebCore/SecurityOriginHash.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class SecurityOrigin;
}

namespace WebKit {

struct WebsiteDataRecord {
    static String displayNameForCookieHostName(const String& hostName);
    static String displayNameForHostName(const String& hostName);

    static String displayNameForOrigin(const WebCore::SecurityOriginData&);

    void add(WebsiteDataType, const WebCore::SecurityOriginData&);
    void addCookieHostName(const String& hostName);
    void addHSTSCacheHostname(const String& hostName);
    void addAlternativeServicesHostname(const String& hostName);
    void addResourceLoadStatisticsRegistrableDomain(const WebCore::RegistrableDomain&);

    bool matches(const WebCore::RegistrableDomain&) const;
    String topPrivatelyControlledDomain();

    WebsiteDataRecord isolatedCopy() const &;
    WebsiteDataRecord isolatedCopy() &&;

    String displayName;
    OptionSet<WebsiteDataType> types;

    struct Size {
        uint64_t totalSize;
        HashMap<unsigned, uint64_t> typeSizes;
    };
    std::optional<Size> size;

    HashSet<WebCore::SecurityOriginData> origins;
    HashSet<String> cookieHostNames;
    HashSet<String> HSTSCacheHostNames;
    HashSet<String> alternativeServicesHostNames;
    HashSet<WebCore::RegistrableDomain> resourceLoadStatisticsRegistrableDomains;
};

}
