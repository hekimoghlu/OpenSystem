/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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

#include "SVGElement.h"
#include "SVGNames.h"
#include "SVGTests.h"
#include "SVGUnitTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class RepaintRectCalculation : bool;

class SVGMaskElement final : public SVGElement, public SVGTests {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGMaskElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGMaskElement);
public:
    static Ref<SVGMaskElement> create(const QualifiedName&, Document&);

    const SVGLengthValue& x() const { return m_x->currentValue(); }
    const SVGLengthValue& y() const { return m_y->currentValue(); }
    const SVGLengthValue& width() const { return m_width->currentValue(); }
    const SVGLengthValue& height() const { return m_height->currentValue(); }
    SVGUnitTypes::SVGUnitType maskUnits() const { return m_maskUnits->currentValue<SVGUnitTypes::SVGUnitType>(); }
    SVGUnitTypes::SVGUnitType maskContentUnits() const { return m_maskContentUnits->currentValue<SVGUnitTypes::SVGUnitType>(); }

    SVGAnimatedLength& xAnimated() { return m_x; }
    SVGAnimatedLength& yAnimated() { return m_y; }
    SVGAnimatedLength& widthAnimated() { return m_width; }
    SVGAnimatedLength& heightAnimated() { return m_height; }
    SVGAnimatedEnumeration& maskUnitsAnimated() { return m_maskUnits; }
    SVGAnimatedEnumeration& maskContentUnitsAnimated() { return m_maskContentUnits; }

    FloatRect calculateMaskContentRepaintRect(RepaintRectCalculation);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGMaskElement, SVGElement, SVGTests>;

private:
    SVGMaskElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;
    void childrenChanged(const ChildChange&) final;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool needsPendingResourceHandling() const final { return false; }
    bool selfHasRelativeLengths() const final { return true; }
    bool supportsFocus() const final { return false; }

    Ref<SVGAnimatedLength> m_x { SVGAnimatedLength::create(this, SVGLengthMode::Width, "-10%"_s) };
    Ref<SVGAnimatedLength> m_y { SVGAnimatedLength::create(this, SVGLengthMode::Height, "-10%"_s) };
    Ref<SVGAnimatedLength> m_width { SVGAnimatedLength::create(this, SVGLengthMode::Width, "120%"_s) };
    Ref<SVGAnimatedLength> m_height { SVGAnimatedLength::create(this, SVGLengthMode::Height, "120%"_s) };
    Ref<SVGAnimatedEnumeration> m_maskUnits { SVGAnimatedEnumeration::create(this, SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) };
    Ref<SVGAnimatedEnumeration> m_maskContentUnits { SVGAnimatedEnumeration::create(this, SVGUnitTypes::SVG_UNIT_TYPE_USERSPACEONUSE) };
};

} // namespace WebCore
