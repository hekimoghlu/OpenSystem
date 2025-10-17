/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
#include "CSSParserContext.h"

#include "CSSPropertyNames.h"
#include "CSSValuePool.h"
#include "DocumentInlines.h"
#include "DocumentLoader.h"
#include "OriginAccessPatterns.h"
#include "Page.h"
#include "Quirks.h"
#include "Settings.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

// https://drafts.csswg.org/css-values/#url-local-url-flag
bool ResolvedURL::isLocalURL() const
{
    return specifiedURLString.startsWith('#');
}

const CSSParserContext& strictCSSParserContext()
{
    static MainThreadNeverDestroyed<CSSParserContext> strictContext(HTMLStandardMode);
    return strictContext;
}

CSSParserContext::CSSParserContext(CSSParserMode mode, const URL& baseURL)
    : baseURL(baseURL)
    , mode(mode)
{
    // FIXME: We should turn all of the features on from their WebCore Settings defaults.
    if (isUASheetBehavior(mode)) {
        cssAppearanceBaseEnabled = true;
        cssTextUnderlinePositionLeftRightEnabled = true;
        lightDarkColorEnabled = true;
        popoverAttributeEnabled = true;
        propertySettings.cssInputSecurityEnabled = true;
        propertySettings.cssCounterStyleAtRulesEnabled = true;
        propertySettings.viewTransitionsEnabled = true;
        thumbAndTrackPseudoElementsEnabled = true;
    }

    StaticCSSValuePool::init();
}

CSSParserContext::CSSParserContext(const Document& document)
{
    *this = document.cssParserContext();
}

CSSParserContext::CSSParserContext(const Document& document, const URL& sheetBaseURL, ASCIILiteral charset)
    : baseURL { sheetBaseURL.isNull() ? document.baseURL() : sheetBaseURL }
    , charset { charset }
    , mode { document.inQuirksMode() ? HTMLQuirksMode : HTMLStandardMode }
    , isHTMLDocument { document.isHTMLDocument() }
    , hasDocumentSecurityOrigin { sheetBaseURL.isNull() || document.protectedSecurityOrigin()->canRequest(baseURL, OriginAccessPatternsForWebProcess::singleton()) }
    , useSystemAppearance { document.settings().useSystemAppearance() }
    , counterStyleAtRuleImageSymbolsEnabled { document.settings().cssCounterStyleAtRuleImageSymbolsEnabled() }
    , springTimingFunctionEnabled { document.settings().springTimingFunctionEnabled() }
#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    , cssTransformStyleSeparatedEnabled { document.settings().cssTransformStyleSeparatedEnabled() }
#endif
    , masonryEnabled { document.settings().masonryEnabled() }
    , cssAppearanceBaseEnabled { document.settings().cssAppearanceBaseEnabled() }
    , cssNestingEnabled { document.settings().cssNestingEnabled() }
    , cssPaintingAPIEnabled { document.settings().cssPaintingAPIEnabled() }
    , cssScopeAtRuleEnabled { document.settings().cssScopeAtRuleEnabled() }
    , cssShapeFunctionEnabled { document.settings().cssShapeFunctionEnabled() }
    , cssStartingStyleAtRuleEnabled { document.settings().cssStartingStyleAtRuleEnabled() }
    , cssStyleQueriesEnabled { document.settings().cssStyleQueriesEnabled() }
    , cssTextUnderlinePositionLeftRightEnabled { document.settings().cssTextUnderlinePositionLeftRightEnabled() }
    , cssBackgroundClipBorderAreaEnabled  { document.settings().cssBackgroundClipBorderAreaEnabled() }
    , cssWordBreakAutoPhraseEnabled { document.settings().cssWordBreakAutoPhraseEnabled() }
    , popoverAttributeEnabled { document.settings().popoverAttributeEnabled() }
    , sidewaysWritingModesEnabled { document.settings().sidewaysWritingModesEnabled() }
    , cssTextWrapPrettyEnabled { document.settings().cssTextWrapPrettyEnabled() }
    , thumbAndTrackPseudoElementsEnabled { document.settings().thumbAndTrackPseudoElementsEnabled() }
#if ENABLE(SERVICE_CONTROLS)
    , imageControlsEnabled { document.settings().imageControlsEnabled() }
#endif
    , colorLayersEnabled { document.settings().cssColorLayersEnabled() }
    , lightDarkColorEnabled { document.settings().cssLightDarkEnabled() }
    , contrastColorEnabled { document.settings().cssContrastColorEnabled() }
    , targetTextPseudoElementEnabled { document.settings().targetTextPseudoElementEnabled() }
    , viewTransitionTypesEnabled { document.settings().viewTransitionsEnabled() && document.settings().viewTransitionTypesEnabled() }
    , cssProgressFunctionEnabled { document.settings().cssProgressFunctionEnabled() }
    , cssMediaProgressFunctionEnabled { document.settings().cssMediaProgressFunctionEnabled() }
    , cssContainerProgressFunctionEnabled { document.settings().cssContainerProgressFunctionEnabled() }
    , cssRandomFunctionEnabled { document.settings().cssRandomFunctionEnabled() }
    , webkitMediaTextTrackDisplayQuirkEnabled { document.quirks().needsWebKitMediaTextTrackDisplayQuirk() }
    , propertySettings { CSSPropertySettings { document.settings() } }
{
}

void add(Hasher& hasher, const CSSParserContext& context)
{
    uint32_t bits = context.isHTMLDocument                  << 0
        | context.hasDocumentSecurityOrigin                 << 1
        | context.isContentOpaque                           << 2
        | context.useSystemAppearance                       << 3
        | context.springTimingFunctionEnabled               << 4
#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
        | context.cssTransformStyleSeparatedEnabled         << 5
#endif
        | context.masonryEnabled                            << 6
        | context.cssAppearanceBaseEnabled                  << 7
        | context.cssNestingEnabled                         << 8
        | context.cssPaintingAPIEnabled                     << 9
        | context.cssScopeAtRuleEnabled                     << 10
        | context.cssShapeFunctionEnabled                   << 11
        | context.cssTextUnderlinePositionLeftRightEnabled  << 12
        | context.cssBackgroundClipBorderAreaEnabled        << 13
        | context.cssWordBreakAutoPhraseEnabled             << 14
        | context.popoverAttributeEnabled                   << 15
        | context.sidewaysWritingModesEnabled               << 16
        | context.cssTextWrapPrettyEnabled                  << 17
        | context.thumbAndTrackPseudoElementsEnabled        << 18
#if ENABLE(SERVICE_CONTROLS)
        | context.imageControlsEnabled                      << 19
#endif
        | context.colorLayersEnabled                        << 20
        | context.lightDarkColorEnabled                     << 21
        | context.contrastColorEnabled                      << 22
        | context.targetTextPseudoElementEnabled            << 23
        | context.viewTransitionTypesEnabled                << 24
        | context.cssProgressFunctionEnabled                << 25
        | context.cssMediaProgressFunctionEnabled           << 26
        | context.cssContainerProgressFunctionEnabled       << 27
        | context.cssRandomFunctionEnabled                  << 28
        | (uint32_t)context.mode                            << 29; // This is multiple bits, so keep it last.
    add(hasher, context.baseURL, context.charset, context.propertySettings, bits);
}

ResolvedURL CSSParserContext::completeURL(const String& string) const
{
    auto result = [&] () -> ResolvedURL {
        // See also Document::completeURL(const String&), but note that CSS always uses UTF-8 for URLs
        if (string.isNull())
            return { };

        if (CSSValue::isCSSLocalURL(string))
            return { string, URL { string } };

        return { string, { baseURL, string } };
    }();

    if (mode == WebVTTMode && !result.resolvedURL.protocolIsData())
        return { };

    return result;
}

bool mayDependOnBaseURL(const ResolvedURL& resolved)
{
    if (resolved.specifiedURLString.isEmpty())
        return false;

    if (CSSValue::isCSSLocalURL(resolved.specifiedURLString))
        return false;

    if (protocolIs(resolved.specifiedURLString, "data"_s))
        return false;

    return true;
}

}
