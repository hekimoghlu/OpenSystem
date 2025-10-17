/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#include "LegacySchemeRegistry.h"

#include <wtf/Lock.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RobinHoodHashSet.h>
#include <wtf/RuntimeApplicationChecks.h>
#include <wtf/URLParser.h>

#if ENABLE(CONTENT_FILTERING)
#include "ContentFilter.h"
#endif
#if USE(QUICK_LOOK)
#include "QuickLook.h"
#endif

namespace WebCore {

// FIXME: URLSchemesMap is a peculiar type name given that it is a set.

static const URLSchemesMap& builtinLocalURLSchemes();
static std::span<const ASCIILiteral> builtinSecureSchemes();
static std::span<const ASCIILiteral> builtinSchemesWithUniqueOrigins();
static std::span<const ASCIILiteral> builtinEmptyDocumentSchemes();
static std::span<const ASCIILiteral> builtinCanDisplayOnlyIfCanRequestSchemes();
static std::span<const ASCIILiteral> builtinCORSEnabledSchemes();

using ASCIILiteralSpanFunction = std::span<const ASCIILiteral> (*)();

static void add(URLSchemesMap& set, ASCIILiteralSpanFunction function)
{
    for (auto& scheme : function())
        set.add(scheme);
}

static NeverDestroyed<URLSchemesMap> makeNeverDestroyedSchemeSet(ASCIILiteralSpanFunction function)
{
    URLSchemesMap set;
    add(set, function);
    return set;
}

static Lock schemeRegistryLock;

static const URLSchemesMap& allBuiltinSchemes()
{
    static NeverDestroyed schemes = [] {
        static constexpr ASCIILiteralSpanFunction functions[] {
            builtinSecureSchemes,
            builtinSchemesWithUniqueOrigins,
            builtinEmptyDocumentSchemes,
            builtinCanDisplayOnlyIfCanRequestSchemes,
            builtinCORSEnabledSchemes,
        };

        // Other misc schemes that the LegacySchemeRegistry doesn't know about.
        static constexpr ASCIILiteral otherSchemes[] = {
            "webkit-fake-url"_s,
#if PLATFORM(MAC)
            "safari-extension"_s,
#endif
#if USE(QUICK_LOOK)
            QLPreviewProtocol,
#endif
#if ENABLE(CONTENT_FILTERING)
            ContentFilter::urlScheme(),
#endif
        };

        URLSchemesMap set;
        {
            Locker locker { schemeRegistryLock };
            for (auto& scheme : builtinLocalURLSchemes())
                set.add(scheme);
        }
        for (auto& function : functions)
            add(set, function);
        for (auto& scheme : otherSchemes)
            set.add(scheme);
        return set;
    }();
    return schemes;
}

static const URLSchemesMap& builtinLocalURLSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static NeverDestroyed schemes = URLSchemesMap {
        "file"_s,
#if PLATFORM(COCOA)
        "applewebdata"_s,
#endif
    };
    return schemes;
}

static URLSchemesMap& localURLSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static NeverDestroyed<URLSchemesMap> localSchemes = builtinLocalURLSchemes();
    return localSchemes;
}

static URLSchemesMap& displayIsolatedURLSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static NeverDestroyed<URLSchemesMap> displayIsolatedSchemes;
    return displayIsolatedSchemes;
}

static std::span<const ASCIILiteral> builtinSecureSchemes()
{
    static constexpr std::array schemes {
        "https"_s,
        "about"_s,
        "data"_s,
        "wss"_s,
#if PLATFORM(GTK) || PLATFORM(WPE)
        "resource"_s,
#endif
#if ENABLE(PDFJS)
        "webkit-pdfjs-viewer"_s,
#endif
    };
    return schemes;
}

static URLSchemesMap& secureSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static auto secureSchemes = makeNeverDestroyedSchemeSet(builtinSecureSchemes);
    return secureSchemes;
}

static std::span<const ASCIILiteral> builtinSchemesWithUniqueOrigins()
{
    static constexpr std::array schemes {
        "about"_s,
        "javascript"_s,
        // This is an intentional difference from the behavior the HTML specification calls for.
        // See https://bugs.webkit.org/show_bug.cgi?id=11885
        "data"_s,
    };
    return schemes;
}

static URLSchemesMap& schemesWithUniqueOrigins() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static auto schemesWithUniqueOrigins = makeNeverDestroyedSchemeSet(builtinSchemesWithUniqueOrigins);
    return schemesWithUniqueOrigins;
}

static std::span<const ASCIILiteral> builtinEmptyDocumentSchemes()
{
    static constexpr std::array schemes { "about"_s };
    return schemes;
}

static URLSchemesMap& emptyDocumentSchemes()
{
    ASSERT(isMainThread());
    static auto emptyDocumentSchemes = makeNeverDestroyedSchemeSet(builtinEmptyDocumentSchemes);
    return emptyDocumentSchemes;
}

static URLSchemesMap& schemesForbiddenFromDomainRelaxation()
{
    ASSERT(isMainThread());
    static NeverDestroyed<URLSchemesMap> schemes;
    return schemes;
}

static std::span<const ASCIILiteral> builtinCanDisplayOnlyIfCanRequestSchemes()
{
    static constexpr std::array schemes { "blob"_s };
    return schemes;
}

static URLSchemesMap& canDisplayOnlyIfCanRequestSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(!isInNetworkProcess());
    ASSERT(schemeRegistryLock.isHeld());
    static auto canDisplayOnlyIfCanRequestSchemes = makeNeverDestroyedSchemeSet(builtinCanDisplayOnlyIfCanRequestSchemes);
    return canDisplayOnlyIfCanRequestSchemes;
}

static URLSchemesMap& notAllowingJavascriptURLsSchemes()
{
    ASSERT(isMainThread());
    static NeverDestroyed<URLSchemesMap> notAllowingJavascriptURLsSchemes;
    return notAllowingJavascriptURLsSchemes;
}

void LegacySchemeRegistry::registerURLSchemeAsLocal(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    localURLSchemes().add(scheme);
}

void LegacySchemeRegistry::removeURLSchemeRegisteredAsLocal(const String& scheme)
{
    Locker locker { schemeRegistryLock };
    if (builtinLocalURLSchemes().contains(scheme))
        return;

    localURLSchemes().remove(scheme);
}

static MemoryCompactRobinHoodHashSet<String>& schemesHandledBySchemeHandler() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static NeverDestroyed<MemoryCompactRobinHoodHashSet<String>> set;
    return set.get();
}

void LegacySchemeRegistry::registerURLSchemeAsHandledBySchemeHandler(const String& scheme)
{
    Locker locker { schemeRegistryLock };
    schemesHandledBySchemeHandler().add(scheme);
}

bool LegacySchemeRegistry::schemeIsHandledBySchemeHandler(StringView scheme)
{
    Locker locker { schemeRegistryLock };
    return schemesHandledBySchemeHandler().contains<StringViewHashTranslator>(scheme);
}

static URLSchemesMap& schemesAllowingDatabaseAccessInPrivateBrowsing()
{
    ASSERT(isMainThread());
    static NeverDestroyed<URLSchemesMap> schemesAllowingDatabaseAccessInPrivateBrowsing;
    return schemesAllowingDatabaseAccessInPrivateBrowsing;
}

static std::span<const ASCIILiteral> builtinCORSEnabledSchemes()
{
    static constexpr std::array schemes { "http"_s, "https"_s };
    return schemes;
}

static URLSchemesMap& CORSEnabledSchemes()
{
    ASSERT(isMainThread());
    // FIXME: http://bugs.webkit.org/show_bug.cgi?id=77160
    static auto schemes = makeNeverDestroyedSchemeSet(builtinCORSEnabledSchemes);
    return schemes;
}

#if ENABLE(PDFJS)
static std::span<const ASCIILiteral> builtinCSPBypassingSchemes()
{
    static constexpr std::array schemes { "webkit-pdfjs-viewer"_s };
    return schemes;
}
#endif

static URLSchemesMap& ContentSecurityPolicyBypassingSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
#if ENABLE(PDFJS)
    static auto schemes = makeNeverDestroyedSchemeSet(builtinCSPBypassingSchemes);
#else
    static NeverDestroyed<URLSchemesMap> schemes;
#endif
    return schemes;
}

static URLSchemesMap& cachePartitioningSchemes() WTF_REQUIRES_LOCK(schemeRegistryLock)
{
    ASSERT(schemeRegistryLock.isHeld());
    static NeverDestroyed<URLSchemesMap> schemes;
    return schemes;
}

static URLSchemesMap& alwaysRevalidatedSchemes()
{
    ASSERT(isMainThread());
    static NeverDestroyed<URLSchemesMap> schemes;
    return schemes;
}

bool LegacySchemeRegistry::shouldTreatURLSchemeAsLocal(StringView scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return localURLSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsNoAccess(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    schemesWithUniqueOrigins().add(scheme);
}

bool LegacySchemeRegistry::shouldTreatURLSchemeAsNoAccess(StringView scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return schemesWithUniqueOrigins().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsDisplayIsolated(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    displayIsolatedURLSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldTreatURLSchemeAsDisplayIsolated(StringView scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return displayIsolatedURLSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsSecure(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    secureSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldTreatURLSchemeAsSecure(StringView scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return secureSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsEmptyDocument(const String& scheme)
{
    if (scheme.isNull())
        return;
    emptyDocumentSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldLoadURLSchemeAsEmptyDocument(StringView scheme)
{
    return !scheme.isNull() && emptyDocumentSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::setDomainRelaxationForbiddenForURLScheme(bool forbidden, const String& scheme)
{
    if (scheme.isNull())
        return;

    if (forbidden)
        schemesForbiddenFromDomainRelaxation().add(scheme);
    else
        schemesForbiddenFromDomainRelaxation().remove(scheme);
}

bool LegacySchemeRegistry::isDomainRelaxationForbiddenForURLScheme(const String& scheme)
{
    return !scheme.isNull() && schemesForbiddenFromDomainRelaxation().contains(scheme);
}

bool LegacySchemeRegistry::canDisplayOnlyIfCanRequest(StringView scheme)
{
    ASSERT(!isInNetworkProcess());
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return canDisplayOnlyIfCanRequestSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerAsCanDisplayOnlyIfCanRequest(const String& scheme)
{
    ASSERT(!isInNetworkProcess());
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    canDisplayOnlyIfCanRequestSchemes().add(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsNotAllowingJavascriptURLs(const String& scheme)
{
    if (scheme.isNull())
        return;
    notAllowingJavascriptURLsSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldTreatURLSchemeAsNotAllowingJavascriptURLs(const String& scheme)
{
    return !scheme.isNull() && notAllowingJavascriptURLsSchemes().contains(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsAllowingDatabaseAccessInPrivateBrowsing(const String& scheme)
{
    if (scheme.isNull())
        return;
    schemesAllowingDatabaseAccessInPrivateBrowsing().add(scheme);
}

bool LegacySchemeRegistry::allowsDatabaseAccessInPrivateBrowsing(const String& scheme)
{
    return !scheme.isNull() && schemesAllowingDatabaseAccessInPrivateBrowsing().contains(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsCORSEnabled(const String& scheme)
{
    ASSERT(!isInNetworkProcess());
    if (scheme.isNull())
        return;
    CORSEnabledSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldTreatURLSchemeAsCORSEnabled(StringView scheme)
{
    ASSERT(!isInNetworkProcess());
    return !scheme.isNull() && CORSEnabledSchemes().contains<StringViewHashTranslator>(scheme);
}

Vector<String> LegacySchemeRegistry::allURLSchemesRegisteredAsCORSEnabled()
{
    ASSERT(!isInNetworkProcess());
    return copyToVector(CORSEnabledSchemes());
}

void LegacySchemeRegistry::registerURLSchemeAsBypassingContentSecurityPolicy(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    ContentSecurityPolicyBypassingSchemes().add(scheme);
}

void LegacySchemeRegistry::removeURLSchemeRegisteredAsBypassingContentSecurityPolicy(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    ContentSecurityPolicyBypassingSchemes().remove(scheme);
}

bool LegacySchemeRegistry::schemeShouldBypassContentSecurityPolicy(StringView scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return ContentSecurityPolicyBypassingSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsAlwaysRevalidated(const String& scheme)
{
    if (scheme.isNull())
        return;
    alwaysRevalidatedSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldAlwaysRevalidateURLScheme(StringView scheme)
{
    return !scheme.isNull() && alwaysRevalidatedSchemes().contains<StringViewHashTranslator>(scheme);
}

void LegacySchemeRegistry::registerURLSchemeAsCachePartitioned(const String& scheme)
{
    if (scheme.isNull())
        return;

    Locker locker { schemeRegistryLock };
    cachePartitioningSchemes().add(scheme);
}

bool LegacySchemeRegistry::shouldPartitionCacheForURLScheme(const String& scheme)
{
    if (scheme.isNull())
        return false;

    Locker locker { schemeRegistryLock };
    return cachePartitioningSchemes().contains(scheme);
}

bool LegacySchemeRegistry::isUserExtensionScheme(StringView scheme)
{
    // FIXME: Remove this once Safari has adopted WKWebViewConfiguration._corsDisablingPatterns
#if PLATFORM(MAC)
    if (scheme == "safari-extension"_s)
        return true;
#else
    UNUSED_PARAM(scheme);
#endif
    return false;
}

bool LegacySchemeRegistry::isBuiltinScheme(const String& scheme)
{
    return !scheme.isNull() && (allBuiltinSchemes().contains(scheme) || WTF::URLParser::isSpecialScheme(scheme));
}

} // namespace WebCore
