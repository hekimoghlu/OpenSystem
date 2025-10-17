/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
#include "MutableStyleProperties.h"

#include "CSSCustomPropertyValue.h"
#include "CSSParser.h"
#include "CSSValuePool.h"
#include "ImmutableStyleProperties.h"
#include "PropertySetCSSStyleDeclaration.h"
#include "StylePropertiesInlines.h"
#include "StylePropertyShorthand.h"

namespace WebCore {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(MutableStyleProperties);

MutableStyleProperties::MutableStyleProperties(CSSParserMode mode)
    : StyleProperties(mode)
{
}

MutableStyleProperties::MutableStyleProperties(Vector<CSSProperty>&& properties)
    : StyleProperties(HTMLStandardMode)
    , m_propertyVector(WTFMove(properties))
{
}

MutableStyleProperties::~MutableStyleProperties() = default;

MutableStyleProperties::MutableStyleProperties(const StyleProperties& other)
    : StyleProperties(other.cssParserMode())
{
    if (auto* mutableProperties = dynamicDowncast<MutableStyleProperties>(other))
        m_propertyVector = mutableProperties->m_propertyVector;
    else {
        m_propertyVector = WTF::map(downcast<ImmutableStyleProperties>(other), [](auto property) {
            return property.toCSSProperty();
        });
    }
}

Ref<MutableStyleProperties> MutableStyleProperties::create()
{
    return adoptRef(*new MutableStyleProperties(HTMLQuirksMode));
}

Ref<MutableStyleProperties> MutableStyleProperties::create(CSSParserMode mode)
{
    return adoptRef(*new MutableStyleProperties(mode));
}

Ref<MutableStyleProperties> MutableStyleProperties::create(Vector<CSSProperty>&& properties)
{
    return adoptRef(*new MutableStyleProperties(WTFMove(properties)));
}

Ref<MutableStyleProperties> MutableStyleProperties::createEmpty()
{
    return adoptRef(*new MutableStyleProperties({ }));
}

Ref<ImmutableStyleProperties> MutableStyleProperties::immutableCopy() const
{
    return ImmutableStyleProperties::create(m_propertyVector.span(), cssParserMode());
}

Ref<ImmutableStyleProperties> MutableStyleProperties::immutableDeduplicatedCopy() const
{
    return ImmutableStyleProperties::createDeduplicating(m_propertyVector.span(), cssParserMode());
}

inline bool MutableStyleProperties::removeShorthandProperty(CSSPropertyID propertyID, String* returnText)
{
    // FIXME: Use serializeShorthandValue here to return the value of the removed shorthand as we do when removing a longhand.
    if (returnText)
        *returnText = String();
    return removeProperties(shorthandForProperty(propertyID).properties());
}

bool MutableStyleProperties::removePropertyAtIndex(int index, String* returnText)
{
    if (index == -1) {
        if (returnText)
            *returnText = String();
        return false;
    }

    if (returnText) {
        auto property = propertyAt(index);
        *returnText = WebCore::serializeLonghandValue(property.id(), *property.value());
    }

    // A more efficient removal strategy would involve marking entries as empty
    // and sweeping them when the vector grows too big.
    m_propertyVector.remove(index);
    return true;
}

inline bool MutableStyleProperties::removeLonghandProperty(CSSPropertyID propertyID, String* returnText)
{
    return removePropertyAtIndex(findPropertyIndex(propertyID), returnText);
}

bool MutableStyleProperties::removeProperty(CSSPropertyID propertyID, String* returnText)
{
    return isLonghand(propertyID) ? removeLonghandProperty(propertyID, returnText) : removeShorthandProperty(propertyID, returnText);
}

bool MutableStyleProperties::removeCustomProperty(const String& propertyName, String* returnText)
{
    return removePropertyAtIndex(findCustomPropertyIndex(propertyName), returnText);
}

bool MutableStyleProperties::setProperty(CSSPropertyID propertyID, const String& value, CSSParserContext parserContext, IsImportant important, bool* didFailParsing)
{
    if (!isExposed(propertyID, &parserContext.propertySettings) && !isInternal(propertyID)) {
        // Allow internal properties as we use them to handle certain DOM-exposed values
        // (e.g. -webkit-font-size-delta from execCommand('FontSizeDelta')).
        return false;
    }

    // Setting the value to an empty string just removes the property in both IE and Gecko.
    // Setting it to null seems to produce less consistent results, but we treat it just the same.
    if (value.isEmpty())
        return removeProperty(propertyID);

    // When replacing an existing property value, this moves the property to the end of the list.
    // Firefox preserves the position, and MSIE moves the property to the beginning.
    parserContext.mode = cssParserMode();
    auto parseResult = CSSParser::parseValue(*this, propertyID, value, important, parserContext);
    if (didFailParsing)
        *didFailParsing = parseResult == CSSParser::ParseResult::Error;
    return parseResult == CSSParser::ParseResult::Changed;
}

bool MutableStyleProperties::setProperty(CSSPropertyID propertyID, const String& value, IsImportant important, bool* didFailParsing)
{
    CSSParserContext parserContext(cssParserMode());
    return setProperty(propertyID, value, parserContext, important, didFailParsing);
}

bool MutableStyleProperties::setCustomProperty(const String& propertyName, const String& value, CSSParserContext parserContext, IsImportant important)
{
    // Setting the value to an empty string just removes the property in both IE and Gecko.
    // Setting it to null seems to produce less consistent results, but we treat it just the same.
    if (value.isEmpty())
        return removeCustomProperty(propertyName);

    // When replacing an existing property value, this moves the property to the end of the list.
    // Firefox preserves the position, and MSIE moves the property to the beginning.
    parserContext.mode = cssParserMode();
    return CSSParser::parseCustomPropertyValue(*this, AtomString { propertyName }, value, important, parserContext) == CSSParser::ParseResult::Changed;
}

void MutableStyleProperties::setProperty(CSSPropertyID propertyID, RefPtr<CSSValue>&& value, IsImportant important)
{
    if (isLonghand(propertyID)) {
        setProperty(CSSProperty(propertyID, WTFMove(value), important));
        return;
    }
    auto shorthand = shorthandForProperty(propertyID);
    removeProperties(shorthand.properties());
    for (auto longhand : shorthand)
        m_propertyVector.append(CSSProperty(longhand, value.copyRef(), important));
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
bool MutableStyleProperties::canUpdateInPlace(const CSSProperty& property, CSSProperty* toReplace) const
{
    // If the property is in a logical property group, we can't just update the value in-place,
    // because afterwards there might be another property of the same group but different mapping logic.
    // In that case the latter might override the former, so setProperty would have no effect.
    CSSPropertyID id = property.id();
    if (CSSProperty::isInLogicalPropertyGroup(id)) {
        ASSERT(toReplace >= m_propertyVector.begin());
        ASSERT(toReplace < m_propertyVector.end());
        for (CSSProperty* it = toReplace + 1; it != m_propertyVector.end(); ++it) {
            if (CSSProperty::areInSameLogicalPropertyGroupWithDifferentMappingLogic(id, it->id()))
                return false;
        }
    }
    return true;
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

bool MutableStyleProperties::setProperty(const CSSProperty& property, CSSProperty* slot)
{
    ASSERT(property.id() == CSSPropertyCustom || isLonghand(property.id()));
    auto* toReplace = slot;
    if (!slot) {
        if (property.id() == CSSPropertyCustom) {
            if (property.value())
                toReplace = findCustomCSSPropertyWithName(downcast<CSSCustomPropertyValue>(*property.value()).name());
        } else
            toReplace = findCSSPropertyWithID(property.id());
    }
    if (toReplace) {
        if (canUpdateInPlace(property, toReplace)) {
            if (*toReplace == property)
                return false;
            *toReplace = property;
            return true;
        }
        m_propertyVector.remove(toReplace - m_propertyVector.begin());
    }
    m_propertyVector.append(property);
    return true;
}

bool MutableStyleProperties::setProperty(CSSPropertyID propertyID, CSSValueID identifier, IsImportant important)
{
    ASSERT(isLonghand(propertyID));
    return setProperty(CSSProperty(propertyID, CSSPrimitiveValue::create(identifier), important));
}

bool MutableStyleProperties::parseDeclaration(const String& styleDeclaration, CSSParserContext context)
{
    auto oldProperties = WTFMove(m_propertyVector);
    m_propertyVector.clear();

    context.mode = cssParserMode();
    CSSParser(context).parseDeclaration(*this, styleDeclaration);

    // We could do better. Just changing property order does not require style invalidation.
    return oldProperties != m_propertyVector;
}

bool MutableStyleProperties::addParsedProperties(const ParsedPropertyVector& properties)
{
    bool anyChanged = false;
    m_propertyVector.reserveCapacity(m_propertyVector.size() + properties.size());
    for (const auto& property : properties) {
        if (addParsedProperty(property))
            anyChanged = true;
    }
    return anyChanged;
}

bool MutableStyleProperties::addParsedProperty(const CSSProperty& property)
{
    if (property.id() == CSSPropertyCustom) {
        if ((property.value() && !customPropertyIsImportant(downcast<CSSCustomPropertyValue>(*property.value()).name())) || property.isImportant())
            return setProperty(property);
        return false;
    }
    return setProperty(property);
}

bool MutableStyleProperties::mergeAndOverrideOnConflict(const StyleProperties& other)
{
    bool changed = false;
    for (auto property : other)
        changed |= addParsedProperty(property.toCSSProperty());
    return changed;
}

void MutableStyleProperties::clear()
{
    m_propertyVector.clear();
}

bool MutableStyleProperties::removeProperties(std::span<const CSSPropertyID> properties)
{
    if (m_propertyVector.isEmpty())
        return false;

    // FIXME: This is always used with static sets and in that case constructing the hash repeatedly is pretty pointless.
    UncheckedKeyHashSet<CSSPropertyID> toRemove;
    toRemove.add(properties.begin(), properties.end());

    return m_propertyVector.removeAllMatching([&toRemove](const CSSProperty& property) {
        return toRemove.contains(property.id());
    }) > 0;
}

int MutableStyleProperties::findPropertyIndex(CSSPropertyID propertyID) const
{
    // Convert here propertyID into an uint16_t to compare it with the metadata's m_propertyID to avoid
    // the compiler converting it to an int multiple times in the loop.
    auto& properties = m_propertyVector;
    uint16_t id = enumToUnderlyingType(propertyID);
    for (int n = m_propertyVector.size() - 1 ; n >= 0; --n) {
        if (properties[n].metadata().m_propertyID == id)
            return n;
    }
    return -1;
}

int MutableStyleProperties::findCustomPropertyIndex(StringView propertyName) const
{
    auto& properties = m_propertyVector;
    for (int n = m_propertyVector.size() - 1 ; n >= 0; --n) {
        if (properties[n].metadata().m_propertyID == CSSPropertyCustom) {
            // We found a custom property. See if the name matches.
            if (!properties[n].value())
                continue;
            if (downcast<CSSCustomPropertyValue>(*properties[n].value()).name() == propertyName)
                return n;
        }
    }
    return -1;
}

CSSProperty* MutableStyleProperties::findCSSPropertyWithID(CSSPropertyID propertyID)
{
    int foundPropertyIndex = findPropertyIndex(propertyID);
    if (foundPropertyIndex == -1)
        return nullptr;
    return &m_propertyVector.at(foundPropertyIndex);
}

CSSProperty* MutableStyleProperties::findCustomCSSPropertyWithName(const String& propertyName)
{
    int foundPropertyIndex = findCustomPropertyIndex(propertyName);
    if (foundPropertyIndex == -1)
        return nullptr;
    return &m_propertyVector.at(foundPropertyIndex);
}

CSSStyleDeclaration& MutableStyleProperties::ensureInlineCSSStyleDeclaration(StyledElement& parentElement)
{
    if (!m_cssomWrapper)
        m_cssomWrapper = makeUniqueWithoutRefCountedCheck<InlineCSSStyleDeclaration>(*this, parentElement);
    ASSERT(m_cssomWrapper->parentElement() == &parentElement);
    return *m_cssomWrapper;
}

}
