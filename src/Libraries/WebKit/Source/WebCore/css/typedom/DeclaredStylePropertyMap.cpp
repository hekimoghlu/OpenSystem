/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#include "DeclaredStylePropertyMap.h"

#include "CSSCustomPropertyValue.h"
#include "CSSStyleRule.h"
#include "CSSStyleSheet.h"
#include "CSSUnparsedValue.h"
#include "Document.h"
#include "MutableStyleProperties.h"
#include "StyleProperties.h"
#include "StylePropertiesInlines.h"
#include "StyleRule.h"

namespace WebCore {

Ref<DeclaredStylePropertyMap> DeclaredStylePropertyMap::create(CSSStyleRule& ownerRule)
{
    return adoptRef(*new DeclaredStylePropertyMap(ownerRule));
}

DeclaredStylePropertyMap::DeclaredStylePropertyMap(CSSStyleRule& ownerRule)
    : m_ownerRule(ownerRule)
{
}

unsigned DeclaredStylePropertyMap::size() const
{
    if (auto* styleRule = this->styleRule())
        return styleRule->properties().propertyCount();
    return 0;
}

auto DeclaredStylePropertyMap::entries(ScriptExecutionContext* context) const -> Vector<StylePropertyMapEntry>
{
    if (!context)
        return { };

    auto* styleRule = this->styleRule();
    if (!styleRule)
        return { };

    auto& document = downcast<Document>(*context);
    return map(styleRule->properties(), [&] (auto propertyReference) {
        return StylePropertyMapEntry { propertyReference.cssName(), reifyValueToVector(RefPtr<CSSValue> { propertyReference.value() }, propertyReference.id(), document) };
    });
}

RefPtr<CSSValue> DeclaredStylePropertyMap::propertyValue(CSSPropertyID propertyID) const
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return nullptr;
    return styleRule->properties().getPropertyCSSValue(propertyID);
}

String DeclaredStylePropertyMap::shorthandPropertySerialization(CSSPropertyID propertyID) const
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return { };
    return styleRule->properties().getPropertyValue(propertyID);
}

RefPtr<CSSValue> DeclaredStylePropertyMap::customPropertyValue(const AtomString& propertyName) const
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return nullptr;
    return styleRule->properties().getCustomPropertyCSSValue(propertyName.string());
}

bool DeclaredStylePropertyMap::setShorthandProperty(CSSPropertyID propertyID, const String& value)
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return false;

    CSSStyleSheet::RuleMutationScope mutationScope(m_ownerRule.get());
    bool didFailParsing = false;
    styleRule->mutableProperties().setProperty(propertyID, value, IsImportant::No, &didFailParsing);
    return !didFailParsing;
}

bool DeclaredStylePropertyMap::setProperty(CSSPropertyID propertyID, Ref<CSSValue>&& value)
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return false;

    CSSStyleSheet::RuleMutationScope mutationScope(m_ownerRule.get());
    bool didFailParsing = false;
    styleRule->mutableProperties().setProperty(propertyID, value->cssText(), IsImportant::No, &didFailParsing);
    return !didFailParsing;
}

bool DeclaredStylePropertyMap::setCustomProperty(Document&, const AtomString& property, Ref<CSSVariableReferenceValue>&& value)
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return false;

    CSSStyleSheet::RuleMutationScope mutationScope(m_ownerRule.get());
    auto customPropertyValue = CSSCustomPropertyValue::createUnresolved(property, WTFMove(value));
    styleRule->mutableProperties().addParsedProperty(CSSProperty(CSSPropertyCustom, WTFMove(customPropertyValue)));
    return true;
}

void DeclaredStylePropertyMap::removeProperty(CSSPropertyID propertyID)
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return;

    CSSStyleSheet::RuleMutationScope mutationScope(m_ownerRule.get());
    styleRule->mutableProperties().removeProperty(propertyID);
}

void DeclaredStylePropertyMap::removeCustomProperty(const AtomString& property)
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return;

    CSSStyleSheet::RuleMutationScope mutationScope(m_ownerRule.get());
    styleRule->mutableProperties().removeCustomProperty(property.string());
}

StyleRule* DeclaredStylePropertyMap::styleRule() const
{
    return m_ownerRule ? &m_ownerRule->styleRule() : nullptr;
}

void DeclaredStylePropertyMap::clear()
{
    auto* styleRule = this->styleRule();
    if (!styleRule)
        return;

    CSSStyleSheet::RuleMutationScope mutationScope(m_ownerRule.get());
    styleRule->mutableProperties().clear();
}

} // namespace WebCore
