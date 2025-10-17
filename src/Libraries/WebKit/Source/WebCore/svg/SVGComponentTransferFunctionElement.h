/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

#include "FEComponentTransfer.h"
#include "NodeName.h"
#include "SVGElement.h"
#include <wtf/SortedArrayMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

template<>
struct SVGPropertyTraits<ComponentTransferType> {
    static unsigned highestEnumValue() { return enumToUnderlyingType(ComponentTransferType::FECOMPONENTTRANSFER_TYPE_GAMMA); }

    static String toString(ComponentTransferType type)
    {
        switch (type) {
        case ComponentTransferType::FECOMPONENTTRANSFER_TYPE_UNKNOWN:
            return emptyString();
        case ComponentTransferType::FECOMPONENTTRANSFER_TYPE_IDENTITY:
            return "identity"_s;
        case ComponentTransferType::FECOMPONENTTRANSFER_TYPE_TABLE:
            return "table"_s;
        case ComponentTransferType::FECOMPONENTTRANSFER_TYPE_DISCRETE:
            return "discrete"_s;
        case ComponentTransferType::FECOMPONENTTRANSFER_TYPE_LINEAR:
            return "linear"_s;
        case ComponentTransferType::FECOMPONENTTRANSFER_TYPE_GAMMA:
            return "gamma"_s;
        }

        ASSERT_NOT_REACHED();
        return emptyString();
    }

    static ComponentTransferType fromString(const String& value)
    {
        static constexpr std::pair<PackedASCIILiteral<uint64_t>, ComponentTransferType> mappings[] = {
            { "discrete"_s, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_DISCRETE },
            { "gamma"_s, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_GAMMA },
            { "identity"_s, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_IDENTITY },
            { "linear"_s, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_LINEAR },
            { "table"_s, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_TABLE }
        };
        static constexpr SortedArrayMap map { mappings };
        return map.get(value, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_UNKNOWN);
    }
};

class SVGComponentTransferFunctionElement : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGComponentTransferFunctionElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGComponentTransferFunctionElement);
public:
    virtual ComponentTransferChannel channel() const = 0;
    ComponentTransferFunction transferFunction() const;

    ComponentTransferType type() const { return m_type->currentValue<ComponentTransferType>(); }
    const SVGNumberList& tableValues() const { return m_tableValues->currentValue(); }
    float slope() const { return m_slope->currentValue(); }
    float intercept() const { return m_intercept->currentValue(); }
    float amplitude() const { return m_amplitude->currentValue(); }
    float exponent() const { return m_exponent->currentValue(); }
    float offset() const { return m_offset->currentValue(); }

    SVGAnimatedEnumeration& typeAnimated() { return m_type; }
    SVGAnimatedNumberList& tableValuesAnimated() { return m_tableValues; }
    SVGAnimatedNumber& slopeAnimated() { return m_slope; }
    SVGAnimatedNumber& interceptAnimated() { return m_intercept; }
    SVGAnimatedNumber& amplitudeAnimated() { return m_amplitude; }
    SVGAnimatedNumber& exponentAnimated() { return m_exponent; }
    SVGAnimatedNumber& offsetAnimated() { return m_offset; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGComponentTransferFunctionElement, SVGElement>;

protected:
    SVGComponentTransferFunctionElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    bool rendererIsNeeded(const RenderStyle&) override { return false; }
    
private:
    Ref<SVGAnimatedEnumeration> m_type { SVGAnimatedEnumeration::create(this, ComponentTransferType::FECOMPONENTTRANSFER_TYPE_IDENTITY) };
    Ref<SVGAnimatedNumberList> m_tableValues { SVGAnimatedNumberList::create(this) };
    Ref<SVGAnimatedNumber> m_slope { SVGAnimatedNumber::create(this, 1) };
    Ref<SVGAnimatedNumber> m_intercept { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_amplitude { SVGAnimatedNumber::create(this, 1) };
    Ref<SVGAnimatedNumber> m_exponent { SVGAnimatedNumber::create(this, 1) };
    Ref<SVGAnimatedNumber> m_offset { SVGAnimatedNumber::create(this) };
};

} // namespace WebCore
