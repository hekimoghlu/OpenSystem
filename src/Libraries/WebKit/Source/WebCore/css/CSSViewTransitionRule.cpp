/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
#include "CSSViewTransitionRule.h"

#include "CSSPropertyParser.h"
#include "CSSStyleSheet.h"
#include "CSSTokenizer.h"
#include "CSSValueList.h"
#include "CSSValuePair.h"
#include "MutableStyleProperties.h"
#include "StyleProperties.h"
#include "StylePropertiesInlines.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

static std::optional<ViewTransitionNavigation> toViewTransitionNavigationEnum(RefPtr<CSSValue> navigation)
{
    if (!navigation || !navigation->isPrimitiveValue())
        return std::nullopt;

    auto& primitiveNavigationValue = downcast<CSSPrimitiveValue>(*navigation);
    ASSERT(primitiveNavigationValue.isValueID());

    if (primitiveNavigationValue.valueID() == CSSValueAuto)
        return ViewTransitionNavigation::Auto;
    return ViewTransitionNavigation::None;
}

StyleRuleViewTransition::StyleRuleViewTransition(Ref<StyleProperties>&& properties)
    : StyleRuleBase(StyleRuleType::ViewTransition)
{
    m_navigation = toViewTransitionNavigationEnum(properties->getPropertyCSSValue(CSSPropertyNavigation));

    if (auto value = properties->getPropertyCSSValue(CSSPropertyTypes)) {
        auto processSingleValue = [&](const CSSValue& currentValue) {
            if (currentValue.isCustomIdent())
                m_types.append(currentValue.customIdent());
        };
        if (auto* list = dynamicDowncast<CSSValueList>(*value)) {
            for (auto& currentValue : *list)
                processSingleValue(currentValue);
        } else
            processSingleValue(*value);
    }
}

Ref<StyleRuleViewTransition> StyleRuleViewTransition::create(Ref<StyleProperties>&& properties)
{
    return adoptRef(*new StyleRuleViewTransition(WTFMove(properties)));
}

StyleRuleViewTransition::~StyleRuleViewTransition() = default;

Ref<CSSViewTransitionRule> CSSViewTransitionRule::create(StyleRuleViewTransition& rule, CSSStyleSheet* sheet)
{
    return adoptRef(*new CSSViewTransitionRule(rule, sheet));
}

CSSViewTransitionRule::CSSViewTransitionRule(StyleRuleViewTransition& viewTransitionRule, CSSStyleSheet* parent)
    : CSSRule(parent)
    , m_viewTransitionRule(viewTransitionRule)
{
}

CSSViewTransitionRule::~CSSViewTransitionRule() = default;

String CSSViewTransitionRule::cssText() const
{
    StringBuilder builder;
    builder.append("@view-transition { "_s);

    if (m_viewTransitionRule->navigation()) {
        builder.append("navigation: "_s);
        if (*m_viewTransitionRule->navigation() == ViewTransitionNavigation::Auto)
            builder.append("auto"_s);
        else
            builder.append("none"_s);
        builder.append("; "_s);
    }

    if (!types().isEmpty())
        builder.append("types:"_s);
    for (auto& type : types()) {
        builder.append(' ');
        builder.append(type);
    }
    if (!types().isEmpty())
        builder.append('}');

    builder.append('}');
    return builder.toString();
}

AtomString CSSViewTransitionRule::navigation() const
{
    if (!m_viewTransitionRule->navigation())
        return emptyAtom();
    if (*m_viewTransitionRule->navigation() == ViewTransitionNavigation::Auto)
        return "auto"_s;
    return "none"_s;
}

void CSSViewTransitionRule::reattach(StyleRuleBase& rule)
{
    m_viewTransitionRule = downcast<StyleRuleViewTransition>(rule);
}

} // namespace WebCore
