/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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

#include "SVGGraphicsElement.h"
#include "SVGUnitTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderObject;

enum class RepaintRectCalculation : bool;

class SVGClipPathElement final : public SVGGraphicsElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGClipPathElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGClipPathElement);
public:
    static Ref<SVGClipPathElement> create(const QualifiedName&, Document&);

    SVGUnitTypes::SVGUnitType clipPathUnits() const { return m_clipPathUnits->currentValue<SVGUnitTypes::SVGUnitType>(); }
    SVGAnimatedEnumeration& clipPathUnitsAnimated() { return m_clipPathUnits; }

    RefPtr<SVGGraphicsElement> shouldApplyPathClipping() const;

    FloatRect calculateClipContentRepaintRect(RepaintRectCalculation);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGClipPathElement, SVGGraphicsElement>;

private:
    SVGClipPathElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;
    void childrenChanged(const ChildChange&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool supportsFocus() const final { return false; }
    bool needsPendingResourceHandling() const final { return false; }

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    Ref<SVGAnimatedEnumeration> m_clipPathUnits { SVGAnimatedEnumeration::create(this, SVGUnitTypes::SVG_UNIT_TYPE_USERSPACEONUSE) };
};

} // namespace WebCore
