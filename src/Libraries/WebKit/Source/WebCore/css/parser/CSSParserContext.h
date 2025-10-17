/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

#include "CSSParserMode.h"
#include "CSSPropertyNames.h"
#include "StyleRuleType.h"
#include <pal/text/TextEncoding.h>
#include <wtf/HashFunctions.h>
#include <wtf/Hasher.h>
#include <wtf/URL.h>

namespace WebCore {

class Document;

struct ResolvedURL {
    String specifiedURLString;
    URL resolvedURL;

    bool isLocalURL() const;
};

inline ResolvedURL makeResolvedURL(URL&& resolvedURL)
{
    auto string = resolvedURL.string();
    return { WTFMove(string), WTFMove(resolvedURL) };
}

inline bool operator==(const ResolvedURL& a, const ResolvedURL& b)
{
    return a.specifiedURLString == b.specifiedURLString && a.resolvedURL == b.resolvedURL;
}

bool mayDependOnBaseURL(const ResolvedURL&);

struct CSSParserContext {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    URL baseURL;
    ASCIILiteral charset;
    CSSParserMode mode { HTMLStandardMode };
    std::optional<StyleRuleType> enclosingRuleType;
    bool isHTMLDocument : 1 { false };

    // This is only needed to support getMatchedCSSRules.
    bool hasDocumentSecurityOrigin : 1 { false };

    bool isContentOpaque : 1 { false };
    bool useSystemAppearance : 1 { false };
    bool shouldIgnoreImportRules : 1 { false };

    // Settings, excluding those affecting properties.
    bool counterStyleAtRuleImageSymbolsEnabled : 1 { false };
    bool springTimingFunctionEnabled : 1 { false };
#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    bool cssTransformStyleSeparatedEnabled : 1 { false };
#endif
    bool masonryEnabled : 1 { false };
    bool cssAppearanceBaseEnabled : 1 { false };
    bool cssNestingEnabled : 1 { false };
    bool cssPaintingAPIEnabled : 1 { false };
    bool cssScopeAtRuleEnabled : 1 { false };
    bool cssShapeFunctionEnabled : 1 { false };
    bool cssStartingStyleAtRuleEnabled : 1 { false };
    bool cssStyleQueriesEnabled : 1 { false };
    bool cssTextUnderlinePositionLeftRightEnabled : 1 { false };
    bool cssBackgroundClipBorderAreaEnabled : 1 { false };
    bool cssWordBreakAutoPhraseEnabled : 1 { false };
    bool popoverAttributeEnabled : 1 { false };
    bool sidewaysWritingModesEnabled : 1 { false };
    bool cssTextWrapPrettyEnabled : 1 { false };
    bool thumbAndTrackPseudoElementsEnabled : 1 { false };
#if ENABLE(SERVICE_CONTROLS)
    bool imageControlsEnabled : 1 { false };
#endif
    bool colorLayersEnabled : 1 { false };
    bool lightDarkColorEnabled : 1 { false };
    bool contrastColorEnabled : 1 { false };
    bool targetTextPseudoElementEnabled : 1 { false };
    bool viewTransitionTypesEnabled : 1 { false };
    bool cssProgressFunctionEnabled : 1 { false };
    bool cssMediaProgressFunctionEnabled : 1 { false };
    bool cssContainerProgressFunctionEnabled : 1 { false };
    bool cssRandomFunctionEnabled : 1 { false };
    bool webkitMediaTextTrackDisplayQuirkEnabled : 1 { false };

    // Settings, those affecting properties.
    CSSPropertySettings propertySettings;

    CSSParserContext(CSSParserMode, const URL& baseURL = URL());
    WEBCORE_EXPORT CSSParserContext(const Document&);
    CSSParserContext(const Document&, const URL& baseURL, ASCIILiteral charset = ""_s);
    ResolvedURL completeURL(const String&) const;

    bool operator==(const CSSParserContext&) const = default;
};

void add(Hasher&, const CSSParserContext&);

WEBCORE_EXPORT const CSSParserContext& strictCSSParserContext();

struct CSSParserContextHash {
    static unsigned hash(const CSSParserContext& context) { return computeHash(context); }
    static bool equal(const CSSParserContext& a, const CSSParserContext& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::CSSParserContext> : GenericHashTraits<WebCore::CSSParserContext> {
    static void constructDeletedValue(WebCore::CSSParserContext& slot) { new (NotNull, &slot.baseURL) URL(WTF::HashTableDeletedValue); }
    static bool isDeletedValue(const WebCore::CSSParserContext& value) { return value.baseURL.isHashTableDeletedValue(); }
    static WebCore::CSSParserContext emptyValue() { return WebCore::CSSParserContext(WebCore::HTMLStandardMode); }
};

template<> struct DefaultHash<WebCore::CSSParserContext> : WebCore::CSSParserContextHash { };

} // namespace WTF
