/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "ComputedStylePropertyMapReadOnly.h"

#include "CSSComputedStyleDeclaration.h"
#include "CSSPropertyParser.h"
#include "Document.h"
#include "DocumentInlines.h"
#include "Element.h"
#include "RenderStyleInlines.h"
#include "StylePropertyShorthand.h"
#include "StyleScope.h"
#include <wtf/KeyValuePair.h>

namespace WebCore {

Ref<ComputedStylePropertyMapReadOnly> ComputedStylePropertyMapReadOnly::create(Element& element)
{
    return adoptRef(*new ComputedStylePropertyMapReadOnly(element));
}

ComputedStylePropertyMapReadOnly::ComputedStylePropertyMapReadOnly(Element& element)
    : m_element(element)
{
}

RefPtr<CSSValue> ComputedStylePropertyMapReadOnly::propertyValue(CSSPropertyID propertyID) const
{
    return ComputedStyleExtractor(m_element.get()).propertyValue(propertyID, ComputedStyleExtractor::UpdateLayout::Yes, ComputedStyleExtractor::PropertyValueType::Computed);
}

String ComputedStylePropertyMapReadOnly::shorthandPropertySerialization(CSSPropertyID propertyID) const
{
    auto value = propertyValue(propertyID);
    return value ? value->cssText() : String();
}

RefPtr<CSSValue> ComputedStylePropertyMapReadOnly::customPropertyValue(const AtomString& property) const
{
    return ComputedStyleExtractor(m_element.get()).customPropertyValue(property);
}

unsigned ComputedStylePropertyMapReadOnly::size() const
{
    // https://drafts.css-houdini.org/css-typed-om-1/#dom-stylepropertymapreadonly-size
    RefPtr element = protectedElement();
    if (!element)
        return 0;

    ComputedStyleExtractor::updateStyleIfNeededForProperty(*element.get(), CSSPropertyCustom);

    auto* style = element->computedStyle();
    if (!style)
        return 0;

    return element->document().exposedComputedCSSPropertyIDs().size() + style->inheritedCustomProperties().size() + style->nonInheritedCustomProperties().size();
}

Vector<StylePropertyMapReadOnly::StylePropertyMapEntry> ComputedStylePropertyMapReadOnly::entries(ScriptExecutionContext*) const
{
    RefPtr element = protectedElement();
    if (!element)
        return { };

    // https://drafts.css-houdini.org/css-typed-om-1/#the-stylepropertymap
    Vector<StylePropertyMapReadOnly::StylePropertyMapEntry> values;

    // Ensure custom property counts are correct.
    ComputedStyleExtractor::updateStyleIfNeededForProperty(*element.get(), CSSPropertyCustom);

    auto* style = element->computedStyle();
    if (!style)
        return values;

    Ref document = element->protectedDocument();
    const auto& inheritedCustomProperties = style->inheritedCustomProperties();
    const auto& nonInheritedCustomProperties = style->nonInheritedCustomProperties();
    const auto& exposedComputedCSSPropertyIDs = document->exposedComputedCSSPropertyIDs();
    values.reserveInitialCapacity(exposedComputedCSSPropertyIDs.size() + inheritedCustomProperties.size() + nonInheritedCustomProperties.size());

    ComputedStyleExtractor extractor { element.get() };
    values.appendContainerWithMapping(exposedComputedCSSPropertyIDs, [&](auto propertyID) {
        auto value = extractor.propertyValue(propertyID, ComputedStyleExtractor::UpdateLayout::No, ComputedStyleExtractor::PropertyValueType::Computed);
        return makeKeyValuePair(nameString(propertyID), StylePropertyMapReadOnly::reifyValueToVector(WTFMove(value), propertyID, document));
    });

    for (auto* map : { &nonInheritedCustomProperties, &inheritedCustomProperties }) {
        map->forEach([&](auto& it) {
            values.append(makeKeyValuePair(it.key, StylePropertyMapReadOnly::reifyValueToVector(const_cast<CSSCustomPropertyValue*>(it.value.get()), std::nullopt, document)));
            return IterationStatus::Continue;
        });
    }

    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
        const auto& nameA = a.key;
        const auto& nameB = b.key;
        if (nameA.startsWith("--"_s))
            return nameB.startsWith("--"_s) && codePointCompareLessThan(nameA, nameB);

        if (nameA.startsWith('-'))
            return nameB.startsWith("--"_s) || (nameB.startsWith('-') && codePointCompareLessThan(nameA, nameB));

        return nameB.startsWith('-') || codePointCompareLessThan(nameA, nameB);
    });

    return values;
}

} // namespace WebCore
