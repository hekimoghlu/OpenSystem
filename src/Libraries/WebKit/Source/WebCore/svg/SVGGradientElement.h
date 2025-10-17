/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 28, 2022.
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

#include "Gradient.h"
#include "SVGElement.h"
#include "SVGNames.h"
#include "SVGURIReference.h"
#include "SVGUnitTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum SVGSpreadMethodType {
    SVGSpreadMethodUnknown = 0,
    SVGSpreadMethodPad,
    SVGSpreadMethodReflect,
    SVGSpreadMethodRepeat
};

template<>
struct SVGPropertyTraits<SVGSpreadMethodType> {
    static unsigned highestEnumValue() { return SVGSpreadMethodRepeat; }

    static String toString(SVGSpreadMethodType type)
    {
        switch (type) {
        case SVGSpreadMethodUnknown:
            return emptyString();
        case SVGSpreadMethodPad:
            return "pad"_s;
        case SVGSpreadMethodReflect:
            return "reflect"_s;
        case SVGSpreadMethodRepeat:
            return "repeat"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static SVGSpreadMethodType fromString(const String& value)
    {
        if (value == "pad"_s)
            return SVGSpreadMethodPad;
        if (value == "reflect"_s)
            return SVGSpreadMethodReflect;
        if (value == "repeat"_s)
            return SVGSpreadMethodRepeat;
        return SVGSpreadMethodUnknown;
    }
};

class SVGGradientElement : public SVGElement, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGGradientElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGGradientElement);
public:
    enum {
        SVG_SPREADMETHOD_UNKNOWN = SVGSpreadMethodUnknown,
        SVG_SPREADMETHOD_PAD = SVGSpreadMethodReflect,
        SVG_SPREADMETHOD_REFLECT = SVGSpreadMethodRepeat,
        SVG_SPREADMETHOD_REPEAT = SVGSpreadMethodUnknown
    };

    GradientColorStops buildStops();

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGGradientElement, SVGElement, SVGURIReference>;

    SVGSpreadMethodType spreadMethod() const { return m_spreadMethod->currentValue<SVGSpreadMethodType>(); }
    SVGUnitTypes::SVGUnitType gradientUnits() const { return m_gradientUnits->currentValue<SVGUnitTypes::SVGUnitType>(); }
    const SVGTransformList& gradientTransform() const { return m_gradientTransform->currentValue(); }

    SVGAnimatedEnumeration& spreadMethodAnimated() { return m_spreadMethod; }
    SVGAnimatedEnumeration& gradientUnitsAnimated() { return m_gradientUnits; }
    SVGAnimatedTransformList& gradientTransformAnimated() { return m_gradientTransform; }

protected:
    SVGGradientElement(const QualifiedName&, Document&, UniqueRef<SVGPropertyRegistry>&&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    void invalidateGradientResource();

private:
    bool needsPendingResourceHandling() const override { return false; }
    void childrenChanged(const ChildChange&) override;

    Ref<SVGAnimatedEnumeration> m_spreadMethod { SVGAnimatedEnumeration::create(this, SVGSpreadMethodPad) };
    Ref<SVGAnimatedEnumeration> m_gradientUnits { SVGAnimatedEnumeration::create(this, SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) };
    Ref<SVGAnimatedTransformList> m_gradientTransform { SVGAnimatedTransformList::create(this) };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SVGGradientElement)
static bool isType(const WebCore::SVGElement& element)
{
    return element.hasTagName(WebCore::SVGNames::radialGradientTag) || element.hasTagName(WebCore::SVGNames::linearGradientTag);
}
static bool isType(const WebCore::Node& node)
{
    auto* svgElement = dynamicDowncast<WebCore::SVGElement>(node);
    return svgElement && isType(*svgElement);
}
SPECIALIZE_TYPE_TRAITS_END()
