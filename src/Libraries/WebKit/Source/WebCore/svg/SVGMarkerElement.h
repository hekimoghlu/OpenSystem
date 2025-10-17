/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "SVGMarkerTypes.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGMarkerElement final : public SVGElement, public SVGFitToViewBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGMarkerElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGMarkerElement);
public:
    // Forward declare enumerations in the W3C naming scheme, for IDL generation.
    enum {
        SVG_MARKERUNITS_UNKNOWN = SVGMarkerUnitsUnknown,
        SVG_MARKERUNITS_USERSPACEONUSE = SVGMarkerUnitsUserSpaceOnUse,
        SVG_MARKERUNITS_STROKEWIDTH = SVGMarkerUnitsStrokeWidth
    };

    enum {
        SVG_MARKER_ORIENT_UNKNOWN = SVGMarkerOrientUnknown,
        SVG_MARKER_ORIENT_AUTO = SVGMarkerOrientAuto,
        SVG_MARKER_ORIENT_ANGLE = SVGMarkerOrientAngle
    };

    static Ref<SVGMarkerElement> create(const QualifiedName&, Document&);

    AffineTransform viewBoxToViewTransform(float viewWidth, float viewHeight) const;

    const SVGLengthValue& refX() const { return m_refX->currentValue(); }
    const SVGLengthValue& refY() const { return m_refY->currentValue(); }
    const SVGLengthValue& markerWidth() const { return m_markerWidth->currentValue(); }
    const SVGLengthValue& markerHeight() const { return m_markerHeight->currentValue(); }
    SVGMarkerUnitsType markerUnits() const { return m_markerUnits->currentValue<SVGMarkerUnitsType>(); }
    const SVGAngleValue& orientAngle() const { return m_orientAngle->currentValue(); }
    SVGMarkerOrientType orientType() const { return m_orientType->currentValue<SVGMarkerOrientType>(); }

    SVGAnimatedLength& refXAnimated() { return m_refX; }
    SVGAnimatedLength& refYAnimated() { return m_refY; }
    SVGAnimatedLength& markerWidthAnimated() { return m_markerWidth; }
    SVGAnimatedLength& markerHeightAnimated() { return m_markerHeight; }
    SVGAnimatedEnumeration& markerUnitsAnimated() { return m_markerUnits; }
    SVGAnimatedAngle& orientAngleAnimated() { return m_orientAngle; }
    Ref<SVGAnimatedEnumeration> orientTypeAnimated() { return m_orientType.copyRef(); }

    AtomString orient() const;
    void setOrient(const AtomString&);

    void setOrientToAuto();
    void setOrientToAngle(const SVGAngle&);

private:
    SVGMarkerElement(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGMarkerElement, SVGElement, SVGFitToViewBox>;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;
    void childrenChanged(const ChildChange&) override;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;
    bool rendererIsNeeded(const RenderStyle&) override { return true; }

    bool needsPendingResourceHandling() const override { return false; }

    bool selfHasRelativeLengths() const override;

    bool supportsFocus() const final { return false; }

    void invalidateMarkerResource();

    Ref<SVGAnimatedLength> m_refX { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_refY { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_markerWidth { SVGAnimatedLength::create(this, SVGLengthMode::Width, "3"_s) };
    Ref<SVGAnimatedLength> m_markerHeight { SVGAnimatedLength::create(this, SVGLengthMode::Height, "3"_s) };
    Ref<SVGAnimatedEnumeration> m_markerUnits { SVGAnimatedEnumeration::create(this, SVGMarkerUnitsStrokeWidth) };
    Ref<SVGAnimatedAngle> m_orientAngle { SVGAnimatedAngle::create(this) };
    Ref<SVGAnimatedOrientType> m_orientType { SVGAnimatedOrientType::create(this, SVGMarkerOrientAngle) };
};

} // namespace WebCore
