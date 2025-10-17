/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#include "JSCSSRule.h"

#include "CSSContainerRule.h"
#include "CSSCounterStyleRule.h"
#include "CSSFontFaceRule.h"
#include "CSSFontFeatureValuesRule.h"
#include "CSSFontPaletteValuesRule.h"
#include "CSSImportRule.h"
#include "CSSKeyframeRule.h"
#include "CSSKeyframesRule.h"
#include "CSSLayerBlockRule.h"
#include "CSSLayerStatementRule.h"
#include "CSSMediaRule.h"
#include "CSSNamespaceRule.h"
#include "CSSNestedDeclarations.h"
#include "CSSPageRule.h"
#include "CSSPositionTryRule.h"
#include "CSSPropertyRule.h"
#include "CSSScopeRule.h"
#include "CSSStartingStyleRule.h"
#include "CSSStyleRule.h"
#include "CSSSupportsRule.h"
#include "CSSViewTransitionRule.h"
#include "JSCSSContainerRule.h"
#include "JSCSSCounterStyleRule.h"
#include "JSCSSFontFaceRule.h"
#include "JSCSSFontFeatureValuesRule.h"
#include "JSCSSFontPaletteValuesRule.h"
#include "JSCSSImportRule.h"
#include "JSCSSKeyframeRule.h"
#include "JSCSSKeyframesRule.h"
#include "JSCSSLayerBlockRule.h"
#include "JSCSSLayerStatementRule.h"
#include "JSCSSMediaRule.h"
#include "JSCSSNamespaceRule.h"
#include "JSCSSNestedDeclarations.h"
#include "JSCSSPageRule.h"
#include "JSCSSPositionTryRule.h"
#include "JSCSSPropertyRule.h"
#include "JSCSSScopeRule.h"
#include "JSCSSStartingStyleRule.h"
#include "JSCSSStyleRule.h"
#include "JSCSSSupportsRule.h"
#include "JSCSSViewTransitionRule.h"
#include "JSNode.h"
#include "JSStyleSheetCustom.h"
#include "WebCoreOpaqueRootInlines.h"


namespace WebCore {
using namespace JSC;

template<typename Visitor>
void JSCSSRule::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, wrapped());
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSCSSRule);

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<CSSRule>&& rule)
{
    switch (rule->styleRuleType()) {
    case StyleRuleType::Style:
        return createWrapper<CSSStyleRule>(globalObject, WTFMove(rule));
    case StyleRuleType::StyleWithNesting:
        return createWrapper<CSSStyleRule>(globalObject, WTFMove(rule));
    case StyleRuleType::NestedDeclarations:
        return createWrapper<CSSNestedDeclarations>(globalObject, WTFMove(rule));
    case StyleRuleType::Media:
        return createWrapper<CSSMediaRule>(globalObject, WTFMove(rule));
    case StyleRuleType::FontFace:
        return createWrapper<CSSFontFaceRule>(globalObject, WTFMove(rule));
    case StyleRuleType::FontPaletteValues:
        return createWrapper<CSSFontPaletteValuesRule>(globalObject, WTFMove(rule));
    case StyleRuleType::FontFeatureValues:
        return createWrapper<CSSFontFeatureValuesRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Page:
        return createWrapper<CSSPageRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Import:
        return createWrapper<CSSImportRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Namespace:
        return createWrapper<CSSNamespaceRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Keyframe:
        return createWrapper<CSSKeyframeRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Keyframes:
        return createWrapper<CSSKeyframesRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Supports:
        return createWrapper<CSSSupportsRule>(globalObject, WTFMove(rule));
    case StyleRuleType::CounterStyle:
        return createWrapper<CSSCounterStyleRule>(globalObject, WTFMove(rule));
    case StyleRuleType::LayerBlock:
        return createWrapper<CSSLayerBlockRule>(globalObject, WTFMove(rule));
    case StyleRuleType::LayerStatement:
        return createWrapper<CSSLayerStatementRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Container:
        return createWrapper<CSSContainerRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Property:
        return createWrapper<CSSPropertyRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Scope:
        return createWrapper<CSSScopeRule>(globalObject, WTFMove(rule));
    case StyleRuleType::StartingStyle:
        return createWrapper<CSSStartingStyleRule>(globalObject, WTFMove(rule));
    case StyleRuleType::ViewTransition:
        return createWrapper<CSSViewTransitionRule>(globalObject, WTFMove(rule));
    case StyleRuleType::PositionTry:
        return createWrapper<CSSPositionTryRule>(globalObject, WTFMove(rule));
    case StyleRuleType::Charset:
    case StyleRuleType::Margin:
    case StyleRuleType::FontFeatureValuesBlock:
        return createWrapper<CSSRule>(globalObject, WTFMove(rule));
    }
    RELEASE_ASSERT_NOT_REACHED();
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, CSSRule& object)
{
    return wrap(lexicalGlobalObject, globalObject, object);
}

} // namespace WebCore
