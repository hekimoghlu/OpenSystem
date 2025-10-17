/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#include "InspectorCSSOMWrappers.h"

#include "CSSContainerRule.h"
#include "CSSImportRule.h"
#include "CSSLayerBlockRule.h"
#include "CSSLayerStatementRule.h"
#include "CSSMediaRule.h"
#include "CSSPrimitiveValue.h"
#include "CSSRule.h"
#include "CSSStyleRule.h"
#include "CSSStyleSheet.h"
#include "CSSSupportsRule.h"
#include "Document.h"
#include "ExtensionStyleSheets.h"
#include "StyleScope.h"
#include "StyleSheetContents.h"
#include "UserAgentStyle.h"

namespace WebCore {
namespace Style {

void InspectorCSSOMWrappers::collectFromStyleSheetIfNeeded(CSSStyleSheet* styleSheet)
{
    if (!m_styleRuleToCSSOMWrapperMap.isEmpty())
        collect(styleSheet);
}

template <class ListType>
void InspectorCSSOMWrappers::collect(ListType* listType)
{
    if (!listType)
        return;
    unsigned size = listType->length();
    for (unsigned i = 0; i < size; ++i) {
        CSSRule* cssRule = listType->item(i);
        if (!cssRule)
            continue;
        
        switch (cssRule->styleRuleType()) {
        case StyleRuleType::Container:
            collect(uncheckedDowncast<CSSContainerRule>(cssRule));
            break;
        case StyleRuleType::Import:
            collect(uncheckedDowncast<CSSImportRule>(*cssRule).styleSheet());
            break;
        case StyleRuleType::LayerBlock:
            collect(uncheckedDowncast<CSSLayerBlockRule>(cssRule));
            break;
        case StyleRuleType::Media:
            collect(uncheckedDowncast<CSSMediaRule>(cssRule));
            break;
        case StyleRuleType::Supports:
            collect(uncheckedDowncast<CSSSupportsRule>(cssRule));
            break;
        case StyleRuleType::Style:
            m_styleRuleToCSSOMWrapperMap.add(&uncheckedDowncast<CSSStyleRule>(*cssRule).styleRule(), uncheckedDowncast<CSSStyleRule>(cssRule));

            // Eagerly collect rules nested in this style rule.
            collect(uncheckedDowncast<CSSStyleRule>(cssRule));
            break;
        default:
            break;
        }
    }
}

void InspectorCSSOMWrappers::collectFromStyleSheetContents(StyleSheetContents* styleSheet)
{
    if (!styleSheet)
        return;
    auto styleSheetWrapper = CSSStyleSheet::create(*styleSheet);
    m_styleSheetCSSOMWrapperSet.add(styleSheetWrapper.copyRef());
    collect(styleSheetWrapper.ptr());
}

void InspectorCSSOMWrappers::collectFromStyleSheets(const Vector<RefPtr<CSSStyleSheet>>& sheets)
{
    for (auto& sheet : sheets)
        collect(sheet.get());
}

void InspectorCSSOMWrappers::maybeCollectFromStyleSheets(const Vector<RefPtr<CSSStyleSheet>>& sheets)
{
    for (auto& sheet : sheets) {
        if (!m_styleSheetCSSOMWrapperSet.contains(sheet.get())) {
            m_styleSheetCSSOMWrapperSet.add(sheet);
            collect(sheet.get());
        }
    }
}

void InspectorCSSOMWrappers::collectDocumentWrappers(ExtensionStyleSheets& extensionStyleSheets)
{
    if (m_styleRuleToCSSOMWrapperMap.isEmpty()) {
        collectFromStyleSheetContents(UserAgentStyle::defaultStyleSheet);
        collectFromStyleSheetContents(UserAgentStyle::quirksStyleSheet);
        collectFromStyleSheetContents(UserAgentStyle::svgStyleSheet);
        collectFromStyleSheetContents(UserAgentStyle::mathMLStyleSheet);
        collectFromStyleSheetContents(UserAgentStyle::horizontalFormControlsStyleSheet);
        collectFromStyleSheetContents(UserAgentStyle::viewTransitionsStyleSheet);
        collectFromStyleSheetContents(UserAgentStyle::htmlSwitchControlStyleSheet);
#if ENABLE(FULLSCREEN_API)
        collectFromStyleSheetContents(UserAgentStyle::fullscreenStyleSheet);
#endif
        collectFromStyleSheetContents(UserAgentStyle::mediaQueryStyleSheet);

        collect(extensionStyleSheets.pageUserSheet());
        collectFromStyleSheets(extensionStyleSheets.injectedUserStyleSheets());
        collectFromStyleSheets(extensionStyleSheets.documentUserStyleSheets());
        collectFromStyleSheets(extensionStyleSheets.injectedAuthorStyleSheets());
        collectFromStyleSheets(extensionStyleSheets.authorStyleSheetsForTesting());
    }
}

void InspectorCSSOMWrappers::collectScopeWrappers(Scope& styleScope)
{
    maybeCollectFromStyleSheets(styleScope.activeStyleSheets());
}

CSSStyleRule* InspectorCSSOMWrappers::getWrapperForRuleInSheets(const StyleRule* rule)
{
    return m_styleRuleToCSSOMWrapperMap.get(rule);
}

} // namespace Style
} // namespace WebCore
