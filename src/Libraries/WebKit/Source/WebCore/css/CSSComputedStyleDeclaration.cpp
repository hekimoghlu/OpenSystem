/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
#include "CSSComputedStyleDeclaration.h"

#include "CSSProperty.h"
#include "CSSPropertyAnimation.h"
#include "CSSPropertyParser.h"
#include "CSSSelector.h"
#include "CSSSelectorParser.h"
#include "CSSValuePool.h"
#include "ComposedTreeAncestorIterator.h"
#include "ComputedStyleExtractor.h"
#include "DeprecatedCSSOMValue.h"
#include "RenderBox.h"
#include "RenderBoxModelObject.h"
#include "RenderStyleInlines.h"
#include "ShorthandSerializer.h"
#include "StylePropertiesInlines.h"
#include "StylePropertyShorthand.h"
#include "StyleScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CSSComputedStyleDeclaration);

CSSComputedStyleDeclaration::CSSComputedStyleDeclaration(Element& element, AllowVisited allowVisited)
    : m_element(element)
    , m_allowVisitedStyle(allowVisited == AllowVisited::Yes)
{
}

CSSComputedStyleDeclaration::CSSComputedStyleDeclaration(Element& element, IsEmpty isEmpty)
    : m_element(element)
    , m_isEmpty(isEmpty == IsEmpty::Yes)
{
}

CSSComputedStyleDeclaration::CSSComputedStyleDeclaration(Element& element, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
    : m_element(element)
    , m_pseudoElementIdentifier(pseudoElementIdentifier)
{
}

CSSComputedStyleDeclaration::~CSSComputedStyleDeclaration() = default;

Ref<CSSComputedStyleDeclaration> CSSComputedStyleDeclaration::create(Element& element, AllowVisited allowVisited)
{
    return adoptRef(*new CSSComputedStyleDeclaration(element, allowVisited));
}

Ref<CSSComputedStyleDeclaration> CSSComputedStyleDeclaration::create(Element& element, const std::optional<Style::PseudoElementIdentifier>& pseudoElementIdentifier)
{
    return adoptRef(*new CSSComputedStyleDeclaration(element, pseudoElementIdentifier));
}

Ref<CSSComputedStyleDeclaration> CSSComputedStyleDeclaration::createEmpty(Element& element)
{
    return adoptRef(*new CSSComputedStyleDeclaration(element, IsEmpty::Yes));
}

String CSSComputedStyleDeclaration::cssText() const
{
    return emptyString();
}

ExceptionOr<void> CSSComputedStyleDeclaration::setCssText(const String&)
{
    return Exception { ExceptionCode::NoModificationAllowedError };
}

// In CSS 2.1 the returned object should actually contain the "used values"
// rather then the "computed values" (despite the name saying otherwise).
//
// See;
// http://www.w3.org/TR/CSS21/cascade.html#used-value
// http://www.w3.org/TR/DOM-Level-2-Style/css.html#CSS-CSSStyleDeclaration
// https://developer.mozilla.org/en-US/docs/Web/API/Window/getComputedStyle#Notes
RefPtr<CSSValue> CSSComputedStyleDeclaration::getPropertyCSSValue(CSSPropertyID propertyID, ComputedStyleExtractor::UpdateLayout updateLayout) const
{
    if (!isExposed(propertyID, settings()) || m_isEmpty)
        return nullptr;
    return ComputedStyleExtractor(m_element.ptr(), m_allowVisitedStyle, m_pseudoElementIdentifier).propertyValue(propertyID, updateLayout);
}

Ref<MutableStyleProperties> CSSComputedStyleDeclaration::copyProperties() const
{
    if (m_isEmpty)
        return MutableStyleProperties::create();
    return ComputedStyleExtractor(m_element.ptr(), m_allowVisitedStyle, m_pseudoElementIdentifier).copyProperties();
}

const Settings* CSSComputedStyleDeclaration::settings() const
{
    return &m_element->document().settings();
}

const FixedVector<CSSPropertyID>& CSSComputedStyleDeclaration::exposedComputedCSSPropertyIDs() const
{
    return m_element->document().exposedComputedCSSPropertyIDs();
}

String CSSComputedStyleDeclaration::getPropertyValue(CSSPropertyID propertyID) const
{
    if (m_isEmpty)
        return emptyString(); // FIXME: Should this be null instead, as it is in StyleProperties::getPropertyValue?

    auto canUseShorthandSerializerForPropertyValue = [&]() {
        switch (propertyID) {
        case CSSPropertyGap:
        case CSSPropertyGridArea:
        case CSSPropertyGridColumn:
        case CSSPropertyGridRow:
        case CSSPropertyGridTemplate:
            return true;
        default:
            return false;
        }
    };
    if (isShorthand(propertyID) && canUseShorthandSerializerForPropertyValue())
        return serializeShorthandValue({ m_element.ptr(), m_allowVisitedStyle, m_pseudoElementIdentifier }, propertyID);

    auto value = getPropertyCSSValue(propertyID);
    if (!value)
        return emptyString(); // FIXME: Should this be null instead, as it is in StyleProperties::getPropertyValue?
    return value->cssText();
}

unsigned CSSComputedStyleDeclaration::length() const
{
    if (m_isEmpty)
        return 0;

    ComputedStyleExtractor::updateStyleIfNeededForProperty(m_element.get(), CSSPropertyCustom);

    auto* style = m_element->computedStyle(m_pseudoElementIdentifier);
    if (!style)
        return 0;

    return exposedComputedCSSPropertyIDs().size() + style->inheritedCustomProperties().size() + style->nonInheritedCustomProperties().size();
}

String CSSComputedStyleDeclaration::item(unsigned i) const
{
    if (m_isEmpty)
        return String();

    if (i >= length())
        return String();

    if (i < exposedComputedCSSPropertyIDs().size())
        return nameString(exposedComputedCSSPropertyIDs().at(i));

    auto* style = m_element->computedStyle(m_pseudoElementIdentifier);
    if (!style)
        return String();

    const auto& inheritedCustomProperties = style->inheritedCustomProperties();

    // FIXME: findKeyAtIndex does a linear search for the property name, so if
    // we are called in a loop over all item indexes, we'll spend quadratic time
    // searching for keys.

    if (i < exposedComputedCSSPropertyIDs().size() + inheritedCustomProperties.size())
        return inheritedCustomProperties.findKeyAtIndex(i - exposedComputedCSSPropertyIDs().size());

    const auto& nonInheritedCustomProperties = style->nonInheritedCustomProperties();
    return nonInheritedCustomProperties.findKeyAtIndex(i - inheritedCustomProperties.size() - exposedComputedCSSPropertyIDs().size());
}

CSSRule* CSSComputedStyleDeclaration::parentRule() const
{
    return nullptr;
}

CSSRule* CSSComputedStyleDeclaration::cssRules() const
{
    return nullptr;
}

RefPtr<DeprecatedCSSOMValue> CSSComputedStyleDeclaration::getPropertyCSSValue(const String& propertyName)
{
    if (m_isEmpty)
        return nullptr;

    if (isCustomPropertyName(propertyName)) {
        auto value = ComputedStyleExtractor(m_element.ptr(), m_allowVisitedStyle, m_pseudoElementIdentifier).customPropertyValue(AtomString { propertyName });
        if (!value)
            return nullptr;
        return value->createDeprecatedCSSOMWrapper(*this);
    }

    CSSPropertyID propertyID = cssPropertyID(propertyName);
    if (!propertyID)
        return nullptr;
    auto value = getPropertyCSSValue(propertyID);
    if (!value)
        return nullptr;
    return value->createDeprecatedCSSOMWrapper(*this);
}

String CSSComputedStyleDeclaration::getPropertyValue(const String &propertyName)
{
    if (m_isEmpty)
        return String();

    if (isCustomPropertyName(propertyName))
        return ComputedStyleExtractor(m_element.ptr(), m_allowVisitedStyle, m_pseudoElementIdentifier).customPropertyText(AtomString { propertyName });

    CSSPropertyID propertyID = cssPropertyID(propertyName);
    if (!propertyID)
        return String();
    return getPropertyValue(propertyID);
}

String CSSComputedStyleDeclaration::getPropertyPriority(const String&)
{
    // All computed styles have a priority of not "important".
    return emptyString();
}

String CSSComputedStyleDeclaration::getPropertyShorthand(const String&)
{
    return emptyString(); // FIXME: Should this sometimes be null instead of empty, to match a normal style declaration?
}

bool CSSComputedStyleDeclaration::isPropertyImplicit(const String&)
{
    return false;
}

ExceptionOr<void> CSSComputedStyleDeclaration::setProperty(const String&, const String&, const String&)
{
    return Exception { ExceptionCode::NoModificationAllowedError };
}

ExceptionOr<String> CSSComputedStyleDeclaration::removeProperty(const String&)
{
    return Exception { ExceptionCode::NoModificationAllowedError };
}
    
String CSSComputedStyleDeclaration::getPropertyValueInternal(CSSPropertyID propertyID)
{
    return getPropertyValue(propertyID);
}

ExceptionOr<void> CSSComputedStyleDeclaration::setPropertyInternal(CSSPropertyID, const String&, IsImportant)
{
    return Exception { ExceptionCode::NoModificationAllowedError };
}

} // namespace WebCore
