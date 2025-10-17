/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#include "InlineStylePropertyMap.h"

#include "CSSCustomPropertyValue.h"
#include "Document.h"
#include "StyleAttributeMutationScope.h"
#include "StylePropertiesInlines.h"
#include "StyledElement.h"

namespace WebCore {

Ref<InlineStylePropertyMap> InlineStylePropertyMap::create(StyledElement& element)
{
    return adoptRef(*new InlineStylePropertyMap(element));
}

InlineStylePropertyMap::InlineStylePropertyMap(StyledElement& element)
    : m_element(element)
{
}

RefPtr<CSSValue> InlineStylePropertyMap::propertyValue(CSSPropertyID propertyID) const
{
    if (auto* inlineStyle = m_element ? m_element->inlineStyle() : nullptr)
        return inlineStyle->getPropertyCSSValue(propertyID);
    return nullptr;
}

String InlineStylePropertyMap::shorthandPropertySerialization(CSSPropertyID propertyID) const
{
    if (auto* inlineStyle = m_element ? m_element->inlineStyle() : nullptr)
        return inlineStyle->getPropertyValue(propertyID);
    return String();
}

RefPtr<CSSValue> InlineStylePropertyMap::customPropertyValue(const AtomString& property) const
{
    if (auto* inlineStyle = m_element ? m_element->inlineStyle() : nullptr)
        return inlineStyle->getCustomPropertyCSSValue(property.string());
    return nullptr;
}

unsigned InlineStylePropertyMap::size() const
{
    auto* inlineStyle = m_element ? m_element->inlineStyle() : nullptr;
    return inlineStyle ? inlineStyle->propertyCount() : 0;
}

auto InlineStylePropertyMap::entries(ScriptExecutionContext* context) const -> Vector<StylePropertyMapEntry>
{
    if (!m_element || !context)
        return { };

    auto* inlineStyle = m_element->inlineStyle();
    if (!inlineStyle)
        return { };

    auto& document = downcast<Document>(*context);
    return map(*inlineStyle, [&document] (auto property) {
        return StylePropertyMapEntry(property.cssName(), reifyValueToVector(RefPtr<CSSValue> { property.value() }, property.id(), document));
    });
}

void InlineStylePropertyMap::removeProperty(CSSPropertyID propertyID)
{
    if (!m_element)
        return;
    StyleAttributeMutationScope mutationScope { m_element.get() };
    if (m_element->removeInlineStyleProperty(propertyID))
        mutationScope.enqueueMutationRecord();
}

bool InlineStylePropertyMap::setShorthandProperty(CSSPropertyID propertyID, const String& value)
{
    if (!m_element)
        return false;
    StyleAttributeMutationScope mutationScope { m_element.get() };
    bool didFailParsing = false;
    m_element->setInlineStyleProperty(propertyID, value, IsImportant::No, &didFailParsing);
    if (!didFailParsing)
        mutationScope.enqueueMutationRecord();
    return !didFailParsing;
}

bool InlineStylePropertyMap::setProperty(CSSPropertyID propertyID, Ref<CSSValue>&& value)
{
    if (!m_element)
        return false;
    StyleAttributeMutationScope mutationScope { m_element.get() };
    bool didFailParsing = false;
    // FIXME: We should be able to validate CSSValues without having to serialize to text and go through the
    // parser. This is inefficient.
    m_element->setInlineStyleProperty(propertyID, value->cssText(), IsImportant::No, &didFailParsing);
    if (!didFailParsing) {
        m_element->setInlineStyleProperty(propertyID, WTFMove(value));
        mutationScope.enqueueMutationRecord();
    }
    return !didFailParsing;
}

bool InlineStylePropertyMap::setCustomProperty(Document&, const AtomString& property, Ref<CSSVariableReferenceValue>&& value)
{
    if (!m_element)
        return false;

    StyleAttributeMutationScope mutationScope { m_element.get() };
    auto customPropertyValue = CSSCustomPropertyValue::createUnresolved(property, WTFMove(value));
    if (m_element->setInlineStyleCustomProperty(WTFMove(customPropertyValue)))
        mutationScope.enqueueMutationRecord();
    return true;
}

void InlineStylePropertyMap::removeCustomProperty(const AtomString& property)
{
    if (!m_element)
        return;
    StyleAttributeMutationScope mutationScope { m_element.get() };
    if (m_element->removeInlineStyleCustomProperty(property))
        mutationScope.enqueueMutationRecord();
}

void InlineStylePropertyMap::clear()
{
    if (!m_element)
        return;
    StyleAttributeMutationScope mutationScope { m_element.get() };
    m_element->removeAllInlineStyleProperties();
    mutationScope.enqueueMutationRecord();
}

} // namespace WebCore
