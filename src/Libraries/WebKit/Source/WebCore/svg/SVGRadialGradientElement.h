/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#include "SVGGradientElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct RadialGradientAttributes;

class SVGRadialGradientElement final : public SVGGradientElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGRadialGradientElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGRadialGradientElement);
public:
    static Ref<SVGRadialGradientElement> create(const QualifiedName&, Document&);

    bool collectGradientAttributes(RadialGradientAttributes&);

    const SVGLengthValue& cx() const { return m_cx->currentValue(); }
    const SVGLengthValue& cy() const { return m_cy->currentValue(); }
    const SVGLengthValue& r() const { return m_r->currentValue(); }
    const SVGLengthValue& fx() const { return m_fx->currentValue(); }
    const SVGLengthValue& fy() const { return m_fy->currentValue(); }
    const SVGLengthValue& fr() const { return m_fr->currentValue(); }

    SVGAnimatedLength& cxAnimated() { return m_cx; }
    SVGAnimatedLength& cyAnimated() { return m_cy; }
    SVGAnimatedLength& rAnimated() { return m_r; }
    SVGAnimatedLength& fxAnimated() { return m_fx; }
    SVGAnimatedLength& fyAnimated() { return m_fy; }
    SVGAnimatedLength& frAnimated() { return m_fr; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGRadialGradientElement, SVGGradientElement>;

private:
    SVGRadialGradientElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;

    bool selfHasRelativeLengths() const override;
    bool supportsFocus() const final { return false; }

    Ref<SVGAnimatedLength> m_cx { SVGAnimatedLength::create(this, SVGLengthMode::Width, "50%"_s) };
    Ref<SVGAnimatedLength> m_cy { SVGAnimatedLength::create(this, SVGLengthMode::Height, "50%"_s) };
    Ref<SVGAnimatedLength> m_r { SVGAnimatedLength::create(this, SVGLengthMode::Other, "50%"_s) };
    Ref<SVGAnimatedLength> m_fx { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_fy { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_fr { SVGAnimatedLength::create(this, SVGLengthMode::Other, "0%"_s) };
};

} // namespace WebCore
