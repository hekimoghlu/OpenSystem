/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "HTMLMeterElement.h"

#include "Attribute.h"
#include "ElementInlines.h"
#include "ElementIterator.h"
#include "HTMLDivElement.h"
#include "HTMLFormElement.h"
#include "HTMLNames.h"
#include "HTMLParserIdioms.h"
#include "HTMLStyleElement.h"
#include "NodeName.h"
#include "Page.h"
#include "RenderMeter.h"
#include "RenderTheme.h"
#include "ShadowRoot.h"
#include "UserAgentParts.h"
#include "UserAgentStyleSheets.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(HTMLMeterElement);

using namespace HTMLNames;

HTMLMeterElement::HTMLMeterElement(const QualifiedName& tagName, Document& document)
    : HTMLElement(tagName, document)
{
    ASSERT(hasTagName(meterTag));
}

HTMLMeterElement::~HTMLMeterElement() = default;

Ref<HTMLMeterElement> HTMLMeterElement::create(const QualifiedName& tagName, Document& document)
{
    Ref<HTMLMeterElement> meter = adoptRef(*new HTMLMeterElement(tagName, document));
    meter->ensureUserAgentShadowRoot();
    return meter;
}

RenderPtr<RenderElement> HTMLMeterElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (!RenderTheme::singleton().supportsMeter(style.usedAppearance()))
        return RenderElement::createFor(*this, WTFMove(style));

    return createRenderer<RenderMeter>(*this, WTFMove(style));
}

bool HTMLMeterElement::childShouldCreateRenderer(const Node& child) const
{
    return !is<RenderMeter>(renderer()) && HTMLElement::childShouldCreateRenderer(child);
}

void HTMLMeterElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    switch (name.nodeName()) {
    case AttributeNames::valueAttr:
    case AttributeNames::minAttr:
    case AttributeNames::maxAttr:
    case AttributeNames::lowAttr:
    case AttributeNames::highAttr:
    case AttributeNames::optimumAttr:
        didElementStateChange();
        break;
    default:
        HTMLElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
        break;
    }
}

double HTMLMeterElement::min() const
{
    return parseHTMLFloatingPointNumberValue(attributeWithoutSynchronization(minAttr), 0);
}

void HTMLMeterElement::setMin(double min)
{
    setAttributeWithoutSynchronization(minAttr, AtomString::number(min));
}

double HTMLMeterElement::max() const
{
    return std::max(parseHTMLFloatingPointNumberValue(attributeWithoutSynchronization(maxAttr), std::max(1.0, min())), min());
}

void HTMLMeterElement::setMax(double max)
{
    setAttributeWithoutSynchronization(maxAttr, AtomString::number(max));
}

double HTMLMeterElement::value() const
{
    double value = parseHTMLFloatingPointNumberValue(attributeWithoutSynchronization(valueAttr), 0);
    return std::min(std::max(value, min()), max());
}

void HTMLMeterElement::setValue(double value)
{
    setAttributeWithoutSynchronization(valueAttr, AtomString::number(value));
}

double HTMLMeterElement::low() const
{
    double low = parseHTMLFloatingPointNumberValue(attributeWithoutSynchronization(lowAttr), min());
    return std::min(std::max(low, min()), max());
}

void HTMLMeterElement::setLow(double low)
{
    setAttributeWithoutSynchronization(lowAttr, AtomString::number(low));
}

double HTMLMeterElement::high() const
{
    double high = parseHTMLFloatingPointNumberValue(attributeWithoutSynchronization(highAttr), max());
    return std::min(std::max(high, low()), max());
}

void HTMLMeterElement::setHigh(double high)
{
    setAttributeWithoutSynchronization(highAttr, AtomString::number(high));
}

double HTMLMeterElement::optimum() const
{
    double optimum = parseHTMLFloatingPointNumberValue(attributeWithoutSynchronization(optimumAttr), (max() + min()) / 2);
    return std::min(std::max(optimum, min()), max());
}

void HTMLMeterElement::setOptimum(double optimum)
{
    setAttributeWithoutSynchronization(optimumAttr, AtomString::number(optimum));
}

HTMLMeterElement::GaugeRegion HTMLMeterElement::gaugeRegion() const
{
    double lowValue = low();
    double highValue = high();
    double theValue = value();
    double optimumValue = optimum();

    if (optimumValue < lowValue) {
        // The optimum range stays under low
        if (theValue <= lowValue)
            return GaugeRegionOptimum;
        if (theValue <= highValue)
            return GaugeRegionSuboptimal;
        return GaugeRegionEvenLessGood;
    }
    
    if (highValue < optimumValue) {
        // The optimum range stays over high
        if (highValue <= theValue)
            return GaugeRegionOptimum;
        if (lowValue <= theValue)
            return GaugeRegionSuboptimal;
        return GaugeRegionEvenLessGood;
    }

    // The optimum range stays between high and low.
    // According to the standard, <meter> never show GaugeRegionEvenLessGood in this case
    // because the value is never less or greater than min or max.
    if (lowValue <= theValue && theValue <= highValue)
        return GaugeRegionOptimum;
    return GaugeRegionSuboptimal;
}

double HTMLMeterElement::valueRatio() const
{
    double min = this->min();
    double max = this->max();
    double value = this->value();

    if (max <= min)
        return 0;
    return (value - min) / (max - min);
}

static void setValueClass(HTMLElement& element, HTMLMeterElement::GaugeRegion gaugeRegion)
{
    switch (gaugeRegion) {
    case HTMLMeterElement::GaugeRegionOptimum:
        element.setAttribute(HTMLNames::classAttr, "optimum"_s);
        element.setUserAgentPart(UserAgentParts::webkitMeterOptimumValue());
        return;
    case HTMLMeterElement::GaugeRegionSuboptimal:
        element.setAttribute(HTMLNames::classAttr, "suboptimum"_s);
        element.setUserAgentPart(UserAgentParts::webkitMeterSuboptimumValue());
        return;
    case HTMLMeterElement::GaugeRegionEvenLessGood:
        element.setAttribute(HTMLNames::classAttr, "even-less-good"_s);
        element.setUserAgentPart(UserAgentParts::webkitMeterEvenLessGoodValue());
        return;
    default:
        ASSERT_NOT_REACHED();
    }
}

void HTMLMeterElement::didElementStateChange()
{
    m_value->setInlineStyleProperty(CSSPropertyInlineSize, valueRatio()*100, CSSUnitType::CSS_PERCENTAGE);
    setValueClass(*m_value, gaugeRegion());

    if (RenderMeter* render = renderMeter())
        render->updateFromElement();
}

RenderMeter* HTMLMeterElement::renderMeter() const
{
    return dynamicDowncast<RenderMeter>(renderer());
}

void HTMLMeterElement::didAddUserAgentShadowRoot(ShadowRoot& root)
{
    ASSERT(!m_value);

    static MainThreadNeverDestroyed<const String> shadowStyle(StringImpl::createWithoutCopying(meterElementShadowUserAgentStyleSheet));

    auto style = HTMLStyleElement::create(HTMLNames::styleTag, document(), false);
    style->setTextContent(String { shadowStyle });
    root.appendChild(WTFMove(style));

    // Pseudos are set to allow author styling.
    auto inner = HTMLDivElement::create(document());
    inner->setIdAttribute("inner"_s);
    inner->setUserAgentPart(UserAgentParts::webkitMeterInnerElement());
    root.appendChild(inner);

    auto bar = HTMLDivElement::create(document());
    bar->setIdAttribute("bar"_s);
    bar->setUserAgentPart(UserAgentParts::webkitMeterBar());
    inner->appendChild(bar);

    m_value = HTMLDivElement::create(document());
    m_value->setIdAttribute("value"_s);
    bar->appendChild(*m_value);

    didElementStateChange();
}

} // namespace
