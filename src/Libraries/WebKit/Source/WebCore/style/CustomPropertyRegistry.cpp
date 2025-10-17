/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#include "CustomPropertyRegistry.h"

#include "CSSCustomPropertyValue.h"
#include "CSSPropertyParser.h"
#include "CSSRegisteredCustomProperty.h"
#include "ComputedStyleDependencies.h"
#include "Document.h"
#include "Element.h"
#include "KeyframeEffect.h"
#include "RenderStyle.h"
#include "StyleBuilder.h"
#include "StyleResolver.h"
#include "StyleScope.h"
#include "WebAnimation.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace Style {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CustomPropertyRegistry);

CustomPropertyRegistry::CustomPropertyRegistry(Scope& scope)
    : m_scope(scope)
    , m_initialValuePrototypeStyle(RenderStyle::createPtr())
{
}

const CSSRegisteredCustomProperty* CustomPropertyRegistry::get(const AtomString& name) const
{
    ASSERT(isCustomPropertyName(name));

    // API wins.
    // https://drafts.css-houdini.org/css-properties-values-api/#determining-registration
    if (auto* property = m_propertiesFromAPI.get(name))
        return property;

    return m_propertiesFromStylesheet.get(name);
}

bool CustomPropertyRegistry::isInherited(const AtomString& name) const
{
    auto* registered = get(name);
    return registered ? registered->inherits : true;
}


bool CustomPropertyRegistry::registerFromAPI(CSSRegisteredCustomProperty&& property)
{
    // First registration wins.
    // https://drafts.css-houdini.org/css-properties-values-api/#determining-registration
    auto success = m_propertiesFromAPI.ensure(property.name, [&] {
        return makeUniqueRef<CSSRegisteredCustomProperty>(WTFMove(property));
    }).isNewEntry;

    if (success) {
        invalidate(property.name);
        m_scope.didChangeStyleSheetEnvironment();
    }

    return success;
}

void CustomPropertyRegistry::registerFromStylesheet(const StyleRuleProperty::Descriptor& descriptor)
{
    auto syntax = CSSCustomPropertySyntax::parse(descriptor.syntax);
    ASSERT(syntax);

    auto& document = m_scope.document();

    RefPtr<CSSCustomPropertyValue> initialValue;
    RefPtr<CSSVariableData> initialValueTokensForViewportUnits;

    if (descriptor.initialValue) {
        auto tokenRange = descriptor.initialValue->tokenRange();

        auto parsedInitialValue = parseInitialValue(document, descriptor.name, *syntax, tokenRange);
        if (!parsedInitialValue)
            return;

        initialValue = parsedInitialValue->first;
        if (parsedInitialValue->second == ViewportUnitDependency::Yes) {
            initialValueTokensForViewportUnits = CSSVariableData::create(tokenRange);
            m_scope.document().setHasStyleWithViewportUnits();
        }
    } else if (syntax->isUniversal()) {
        // "If the value of the syntax descriptor is the universal syntax definition, then the initial-value descriptor is optional.
        //  If omitted, the initial value of the property is the guaranteed-invalid value."
        // https://drafts.css-houdini.org/css-properties-values-api/#initial-value-descriptor
        initialValue = CSSCustomPropertyValue::createWithID(descriptor.name, CSSValueInvalid);
    } else {
        // Should have been discarded at parse-time.
        ASSERT_NOT_REACHED();
        return;
    }

    auto property = CSSRegisteredCustomProperty {
        AtomString { descriptor.name },
        *syntax,
        *descriptor.inherits,
        WTFMove(initialValue),
        WTFMove(initialValueTokensForViewportUnits)
    };

    // Last rule wins.
    // https://drafts.css-houdini.org/css-properties-values-api/#determining-registration
    m_propertiesFromStylesheet.set(property.name, makeUniqueRef<CSSRegisteredCustomProperty>(WTFMove(property)));

    invalidate(property.name);
}

void CustomPropertyRegistry::clearRegisteredFromStylesheets()
{
    if (m_propertiesFromStylesheet.isEmpty())
        return;

    m_propertiesFromStylesheet.clear();
    invalidate(nullAtom());
}

const RenderStyle& CustomPropertyRegistry::initialValuePrototypeStyle() const
{
    if (m_hasInvalidPrototypeStyle) {
        m_hasInvalidPrototypeStyle = false;

        auto oldStyle = std::exchange(m_initialValuePrototypeStyle, RenderStyle::createPtr());

        auto initializeToStyle = [&](auto& map) {
            for (auto& property : map.values()) {
                if (property->initialValue && !property->initialValue->isInvalid())
                    m_initialValuePrototypeStyle->setCustomPropertyValue(*property->initialValue, property->inherits);
            }
        };

        initializeToStyle(m_propertiesFromStylesheet);
        initializeToStyle(m_propertiesFromAPI);

        m_initialValuePrototypeStyle->deduplicateCustomProperties(*oldStyle);
    }

    return *m_initialValuePrototypeStyle;
}

void CustomPropertyRegistry::invalidate(const AtomString& customProperty)
{
    m_hasInvalidPrototypeStyle = true;
    // Changing property registration may affect computed property values in the cache.
    m_scope.invalidateMatchedDeclarationsCache();

    // FIXME: Invalidate all?
    if (!customProperty.isNull())
        notifyAnimationsOfCustomPropertyRegistration(customProperty);
}

void CustomPropertyRegistry::notifyAnimationsOfCustomPropertyRegistration(const AtomString& customProperty)
{
    auto& document = m_scope.document();
    for (auto* animation : WebAnimation::instances()) {
        if (auto* keyframeEffect = dynamicDowncast<KeyframeEffect>(animation->effect())) {
            if (auto* target = keyframeEffect->target()) {
                if (&target->document() == &document)
                    keyframeEffect->customPropertyRegistrationDidChange(customProperty);
            }
        }
    }
}

auto CustomPropertyRegistry::parseInitialValue(const Document& document, const AtomString& propertyName, const CSSCustomPropertySyntax& syntax, CSSParserTokenRange tokenRange) -> Expected<std::pair<RefPtr<CSSCustomPropertyValue>, ViewportUnitDependency>, ParseInitialValueError>
{
    // FIXME: This parses twice.
    auto dependencies = CSSPropertyParser::collectParsedCustomPropertyValueDependencies(syntax, tokenRange, document.cssParserContext());
    if (!dependencies.isComputationallyIndependent())
        return makeUnexpected(ParseInitialValueError::NotComputationallyIndependent);

    // We don't need to provide a real context style since only computationally independent values are allowed (no 'em' etc).
    auto placeholderStyle = RenderStyle::create();
    Style::Builder builder { placeholderStyle, { document, RenderStyle::defaultStyle() }, { }, { } };

    auto initialValue = CSSPropertyParser::parseTypedCustomPropertyInitialValue(propertyName, syntax, tokenRange, builder.state(), { document });
    if (!initialValue)
        return makeUnexpected(ParseInitialValueError::DidNotParse);

    return std::pair { WTFMove(initialValue), dependencies.viewportDimensions ? ViewportUnitDependency::Yes : ViewportUnitDependency::No };
}

bool CustomPropertyRegistry::invalidatePropertiesWithViewportUnits(Document& document)
{
    bool invalidatedAny = false;

    auto invalidatePropertiesWithViewportUnits = [&](auto& map) {
        for (auto& property : map.values()) {
            if (!property->initialValueTokensForViewportUnits)
                continue;

            auto tokenRange = property->initialValueTokensForViewportUnits->tokenRange();
            auto parsedInitialValue = parseInitialValue(document, property->name, property->syntax, tokenRange);
            if (!parsedInitialValue) {
                ASSERT_NOT_REACHED();
                continue;
            }
            property->initialValue = WTFMove(parsedInitialValue->first);
            ASSERT(parsedInitialValue->second == ViewportUnitDependency::Yes);

            invalidate(property->name);
            invalidatedAny = true;
        }
    };

    invalidatePropertiesWithViewportUnits(m_propertiesFromAPI);
    invalidatePropertiesWithViewportUnits(m_propertiesFromStylesheet);

    return invalidatedAny;
}

}
}
