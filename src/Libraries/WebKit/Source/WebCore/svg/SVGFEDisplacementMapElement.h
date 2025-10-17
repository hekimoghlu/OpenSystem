/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#pragma once

#include "FEDisplacementMap.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
 
template<>
struct SVGPropertyTraits<ChannelSelectorType> {
    static unsigned highestEnumValue() { return enumToUnderlyingType(ChannelSelectorType::CHANNEL_A); }

    static String toString(ChannelSelectorType type)
    {
        switch (type) {
        case ChannelSelectorType::CHANNEL_UNKNOWN:
            return emptyString();
        case ChannelSelectorType::CHANNEL_R:
            return "R"_s;
        case ChannelSelectorType::CHANNEL_G:
            return "G"_s;
        case ChannelSelectorType::CHANNEL_B:
            return "B"_s;
        case ChannelSelectorType::CHANNEL_A:
            return "A"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static ChannelSelectorType fromString(const String& value)
    {
        if (value == "R"_s)
            return ChannelSelectorType::CHANNEL_R;
        if (value == "G"_s)
            return ChannelSelectorType::CHANNEL_G;
        if (value == "B"_s)
            return ChannelSelectorType::CHANNEL_B;
        if (value == "A"_s)
            return ChannelSelectorType::CHANNEL_A;
        return ChannelSelectorType::CHANNEL_UNKNOWN;
    }
};

class SVGFEDisplacementMapElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEDisplacementMapElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEDisplacementMapElement);
public:
    static Ref<SVGFEDisplacementMapElement> create(const QualifiedName&, Document&);

    static ChannelSelectorType stringToChannel(const String&);

    String in1() const { return m_in1->currentValue(); }
    String in2() const { return m_in2->currentValue(); }
    ChannelSelectorType xChannelSelector() const { return m_xChannelSelector->currentValue<ChannelSelectorType>(); }
    ChannelSelectorType yChannelSelector() const { return m_yChannelSelector->currentValue<ChannelSelectorType>(); }
    float scale() const { return m_scale->currentValue(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedString& in2Animated() { return m_in2; }
    SVGAnimatedEnumeration& xChannelSelectorAnimated() { return m_xChannelSelector; }
    SVGAnimatedEnumeration& yChannelSelectorAnimated() { return m_yChannelSelector; }
    SVGAnimatedNumber& scaleAnimated() { return m_scale; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEDisplacementMapElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEDisplacementMapElement(const QualifiedName& tagName, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName& attrName) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() }, AtomString { in2() } }; }
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedString> m_in2 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedEnumeration> m_xChannelSelector { SVGAnimatedEnumeration::create(this, ChannelSelectorType::CHANNEL_A) };
    Ref<SVGAnimatedEnumeration> m_yChannelSelector { SVGAnimatedEnumeration::create(this, ChannelSelectorType::CHANNEL_A) };
    Ref<SVGAnimatedNumber> m_scale { SVGAnimatedNumber::create(this) };
};

} // namespace WebCore
