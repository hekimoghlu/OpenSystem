/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#include "LightSource.h"
#include "SVGElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGFilter;

class SVGFELightElement : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGFELightElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGFELightElement);
public:
    virtual Ref<LightSource> lightSource() const = 0;
    static SVGFELightElement* findLightElement(const SVGElement*);

    float azimuth() const { return m_azimuth->currentValue(); }
    float elevation() const { return m_elevation->currentValue(); }
    float x() const { return m_x->currentValue(); }
    float y() const { return m_y->currentValue(); }
    float z() const { return m_z->currentValue(); }
    float pointsAtX() const { return m_pointsAtX->currentValue(); }
    float pointsAtY() const { return m_pointsAtY->currentValue(); }
    float pointsAtZ() const { return m_pointsAtZ->currentValue(); }
    float specularExponent() const { return m_specularExponent->currentValue(); }
    float limitingConeAngle() const { return m_limitingConeAngle->currentValue(); }

    SVGAnimatedNumber& azimuthAnimated() { return m_azimuth; }
    SVGAnimatedNumber& elevationAnimated() { return m_elevation; }
    SVGAnimatedNumber& xAnimated() { return m_x; }
    SVGAnimatedNumber& yAnimated() { return m_y; }
    SVGAnimatedNumber& zAnimated() { return m_z; }
    SVGAnimatedNumber& pointsAtXAnimated() { return m_pointsAtX; }
    SVGAnimatedNumber& pointsAtYAnimated() { return m_pointsAtY; }
    SVGAnimatedNumber& pointsAtZAnimated() { return m_pointsAtZ; }
    SVGAnimatedNumber& specularExponentAnimated() { return m_specularExponent; }
    SVGAnimatedNumber& limitingConeAngleAnimated() { return m_limitingConeAngle; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGFELightElement, SVGElement>;

protected:
    SVGFELightElement(const QualifiedName&, Document&);

    bool rendererIsNeeded(const RenderStyle&) override { return false; }

private:
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;
    void childrenChanged(const ChildChange&) override;

    Ref<SVGAnimatedNumber> m_azimuth { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_elevation { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_x { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_y { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_z { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_pointsAtX { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_pointsAtY { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_pointsAtZ { SVGAnimatedNumber::create(this) };
    Ref<SVGAnimatedNumber> m_specularExponent { SVGAnimatedNumber::create(this, 1) };
    Ref<SVGAnimatedNumber> m_limitingConeAngle { SVGAnimatedNumber::create(this) };
};

} // namespace WebCore
