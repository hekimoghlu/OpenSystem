/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 1, 2024.
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

#include "GraphicsTypes.h"
#include "SVGFilterPrimitiveStandardAttributes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template<>
struct SVGPropertyTraits<BlendMode> {
    static unsigned highestEnumValue() { return static_cast<unsigned>(BlendMode::Luminosity); }

    static BlendMode fromString(const String& string)
    {
        BlendMode mode = BlendMode::Normal;
        parseBlendMode(string, mode);
        return mode;
    }

    static String toString(BlendMode type)
    {
        if (type < BlendMode::PlusDarker)
            return blendModeName(type);

        return emptyString();
    }
};

class SVGFEBlendElement final : public SVGFilterPrimitiveStandardAttributes {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFEBlendElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFEBlendElement);
public:
    static Ref<SVGFEBlendElement> create(const QualifiedName&, Document&);

    String in1() const { return m_in1->currentValue(); }
    String in2() const { return m_in2->currentValue(); }
    BlendMode mode() const { return m_mode->currentValue<BlendMode>(); }

    SVGAnimatedString& in1Animated() { return m_in1; }
    SVGAnimatedString& in2Animated() { return m_in2; }
    SVGAnimatedEnumeration& modeAnimated() { return m_mode; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFEBlendElement, SVGFilterPrimitiveStandardAttributes>;

private:
    SVGFEBlendElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool setFilterEffectAttribute(FilterEffect&, const QualifiedName& attrName) override;
    Vector<AtomString> filterEffectInputsNames() const override { return { AtomString { in1() }, AtomString { in2() } }; }
    RefPtr<FilterEffect> createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const override;

    Ref<SVGAnimatedString> m_in1 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedString> m_in2 { SVGAnimatedString::create(this) };
    Ref<SVGAnimatedEnumeration> m_mode { SVGAnimatedEnumeration::create(this, BlendMode::Normal) };
};

} // namespace WebCore
