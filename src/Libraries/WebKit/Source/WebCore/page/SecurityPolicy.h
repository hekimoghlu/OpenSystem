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
#pragma once

#include "FetchOptions.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class OriginAccessPatterns;
class SecurityOrigin;
class UserContentURLPattern;

class SecurityPolicy {
public:
    // True if the referrer should be omitted according to ReferrerPolicy::Default.
    // If you intend to send a referrer header, you should use generateReferrerHeader instead.
    WEBCORE_EXPORT static bool shouldHideReferrer(const URL&, const URL& referrer);

    // Returns the referrer's security origin plus a / to make it a canonical URL
    // and thus useable as referrer.
    static String referrerToOriginString(const URL& referrer);

    // Returns the referrer modified according to the referrer policy for a
    // navigation to a given URL. If the referrer returned is empty, the
    // referrer header should be omitted.
    WEBCORE_EXPORT static String generateReferrerHeader(ReferrerPolicy, const URL&, const URL& referrer, const OriginAccessPatterns&);

    static String generateOriginHeader(ReferrerPolicy, const URL&, const SecurityOrigin&, const OriginAccessPatterns&);

    WEBCORE_EXPORT static bool shouldInheritSecurityOriginFromOwner(const URL&);

    static bool isBaseURLSchemeAllowed(const URL&);

    enum LocalLoadPolicy {
        AllowLocalLoadsForAll, // No restriction on local loads.
        AllowLocalLoadsForLocalAndSubstituteData,
        AllowLocalLoadsForLocalOnly,
    };

    WEBCORE_EXPORT static void setLocalLoadPolicy(LocalLoadPolicy);
    static bool restrictAccessToLocal();
    static bool allowSubstituteDataAccessToLocal();

    WEBCORE_EXPORT static void addOriginAccessAllowlistEntry(const SecurityOrigin& sourceOrigin, const String& destinationProtocol, const String& destinationDomain, bool allowDestinationSubdomains);
    WEBCORE_EXPORT static void removeOriginAccessAllowlistEntry(const SecurityOrigin& sourceOrigin, const String& destinationProtocol, const String& destinationDomain, bool allowDestinationSubdomains);
    WEBCORE_EXPORT static void resetOriginAccessAllowlists();

    static bool isAccessAllowed(const SecurityOrigin& activeOrigin, const SecurityOrigin& targetOrigin, const URL& targetURL, const OriginAccessPatterns&);
    static bool isAccessAllowed(const SecurityOrigin& activeOrigin, const URL& targetURL, const OriginAccessPatterns&);
};

} // namespace WebCore
