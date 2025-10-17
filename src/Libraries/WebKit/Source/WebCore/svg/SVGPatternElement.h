/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include "SVGFitToViewBox.h"
#include "SVGNames.h"
#include "SVGTests.h"
#include "SVGURIReference.h"
#include "SVGUnitTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct PatternAttributes;
 
class SVGPatternElement final : public SVGElement, public SVGFitToViewBox, public SVGTests, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGPatternElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGPatternElement);
public:
    static Ref<SVGPatternElement> create(const QualifiedName&, Document&);

    void collectPatternAttributes(PatternAttributes&) const;

    AffineTransform localCoordinateSpaceTransform(SVGLocatable::CTMScope) const final;

    const SVGLengthValue& x() const { return m_x->currentValue(); }
    const SVGLengthValue& y() const { return m_y->currentValue(); }
    const SVGLengthValue& width() const { return m_width->currentValue(); }
    const SVGLengthValue& height() const { return m_height->currentValue(); }
    SVGUnitTypes::SVGUnitType patternUnits() const { return m_patternUnits->currentValue<SVGUnitTypes::SVGUnitType>(); }
    SVGUnitTypes::SVGUnitType patternContentUnits() const { return m_patternContentUnits->currentValue<SVGUnitTypes::SVGUnitType>(); }
    const SVGTransformList& patternTransform() const { return m_patternTransform->currentValue(); }
    Ref<const SVGTransformList> protectedPatternTransform() const;

    SVGAnimatedLength& xAnimated() { return m_x; }
    SVGAnimatedLength& yAnimated() { return m_y; }
    SVGAnimatedLength& widthAnimated() { return m_width; }
    SVGAnimatedLength& heightAnimated() { return m_height; }
    SVGAnimatedEnumeration& patternUnitsAnimated() { return m_patternUnits; }
    SVGAnimatedEnumeration& patternContentUnitsAnimated() { return m_patternContentUnits; }
    SVGAnimatedTransformList& patternTransformAnimated() { return m_patternTransform; }

private:
    SVGPatternElement(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGPatternElement, SVGElement, SVGFitToViewBox, SVGTests, SVGURIReference>;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;
    void childrenChanged(const ChildChange&) final;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool needsPendingResourceHandling() const final { return false; }
    bool selfHasRelativeLengths() const final { return true; }
    bool supportsFocus() const final { return false; }

    Ref<SVGAnimatedLength> m_x { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_width { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_height { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedEnumeration> m_patternUnits { SVGAnimatedEnumeration::create(this, SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) };
    Ref<SVGAnimatedEnumeration> m_patternContentUnits { SVGAnimatedEnumeration::create(this, SVGUnitTypes::SVG_UNIT_TYPE_USERSPACEONUSE) };
    Ref<SVGAnimatedTransformList> m_patternTransform { SVGAnimatedTransformList::create(this) };
};

} // namespace WebCore
