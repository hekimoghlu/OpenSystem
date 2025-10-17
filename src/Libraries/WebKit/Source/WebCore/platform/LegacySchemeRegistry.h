/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include <wtf/HashSet.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// FIXME: Make UncheckedKeyHashSet<String>::contains(StringView) work and use StringViews here.
using URLSchemesMap = UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash>;

class LegacySchemeRegistry {
public:
    WEBCORE_EXPORT static void registerURLSchemeAsLocal(const String&); // Thread safe.
    static void removeURLSchemeRegisteredAsLocal(const String&); // Thread safe.

    WEBCORE_EXPORT static bool shouldTreatURLSchemeAsLocal(StringView); // Thread safe.
    WEBCORE_EXPORT static bool isBuiltinScheme(const String&);
    
    // Secure schemes do not trigger mixed content warnings. For example,
    // https and data are secure schemes because they cannot be corrupted by
    // active network attackers.
    WEBCORE_EXPORT static void registerURLSchemeAsSecure(const String&); // Thread safe.
    static bool shouldTreatURLSchemeAsSecure(StringView); // Thread safe.

    WEBCORE_EXPORT static void registerURLSchemeAsNoAccess(const String&); // Thread safe.
    static bool shouldTreatURLSchemeAsNoAccess(StringView); // Thread safe.

    // Display-isolated schemes can only be displayed (in the sense of
    // SecurityOrigin::canDisplay) by documents from the same scheme.
    WEBCORE_EXPORT static void registerURLSchemeAsDisplayIsolated(const String&); // Thread safe.
    static bool shouldTreatURLSchemeAsDisplayIsolated(StringView); // Thread safe.

    WEBCORE_EXPORT static void registerURLSchemeAsEmptyDocument(const String&);
    WEBCORE_EXPORT static bool shouldLoadURLSchemeAsEmptyDocument(StringView);

    WEBCORE_EXPORT static void setDomainRelaxationForbiddenForURLScheme(bool forbidden, const String&);
    static bool isDomainRelaxationForbiddenForURLScheme(const String&);

    // Such schemes should delegate to SecurityOrigin::canRequest for any URL
    // passed to SecurityOrigin::canDisplay.
    static bool canDisplayOnlyIfCanRequest(StringView scheme); // Thread safe.
    WEBCORE_EXPORT static void registerAsCanDisplayOnlyIfCanRequest(const String& scheme); // Thread safe.

    // Schemes against which javascript: URLs should not be allowed to run (stop
    // bookmarklets from running on sensitive pages). 
    static void registerURLSchemeAsNotAllowingJavascriptURLs(const String& scheme);
    static bool shouldTreatURLSchemeAsNotAllowingJavascriptURLs(const String& scheme);

    // Let some schemes opt-out of Private Browsing's default behavior of prohibiting read/write
    // access to Local Storage and Databases.
    WEBCORE_EXPORT static void registerURLSchemeAsAllowingDatabaseAccessInPrivateBrowsing(const String& scheme);
    static bool allowsDatabaseAccessInPrivateBrowsing(const String& scheme);

    // Allow non-HTTP schemes to be registered to allow CORS requests.
    WEBCORE_EXPORT static void registerURLSchemeAsCORSEnabled(const String& scheme);
    WEBCORE_EXPORT static bool shouldTreatURLSchemeAsCORSEnabled(StringView scheme);
    WEBCORE_EXPORT static Vector<String> allURLSchemesRegisteredAsCORSEnabled();

    WEBCORE_EXPORT static void registerURLSchemeAsHandledBySchemeHandler(const String&);
    WEBCORE_EXPORT static bool schemeIsHandledBySchemeHandler(StringView);

    // Allow resources from some schemes to load on a page, regardless of its
    // Content Security Policy.
    WEBCORE_EXPORT static void registerURLSchemeAsBypassingContentSecurityPolicy(const String& scheme); // Thread safe.
    WEBCORE_EXPORT static void removeURLSchemeRegisteredAsBypassingContentSecurityPolicy(const String& scheme); // Thread safe.
    static bool schemeShouldBypassContentSecurityPolicy(StringView scheme); // Thread safe.

    // Schemes whose responses should always be revalidated.
    WEBCORE_EXPORT static void registerURLSchemeAsAlwaysRevalidated(const String&);
    static bool shouldAlwaysRevalidateURLScheme(StringView);

    // Schemes whose requests should be partitioned in the cache
    WEBCORE_EXPORT static void registerURLSchemeAsCachePartitioned(const String& scheme); // Thread safe.
    static bool shouldPartitionCacheForURLScheme(const String& scheme); // Thread safe.

    static bool isUserExtensionScheme(StringView scheme);
};

} // namespace WebCore
