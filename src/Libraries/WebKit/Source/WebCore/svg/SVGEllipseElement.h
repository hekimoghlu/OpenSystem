/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

#include "SVGGeometryElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGEllipseElement final : public SVGGeometryElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGEllipseElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGEllipseElement);
public:
    static Ref<SVGEllipseElement> create(const QualifiedName&, Document&);

    const SVGLengthValue& cx() const { return m_cx->currentValue(); }
    const SVGLengthValue& cy() const { return m_cy->currentValue(); }
    const SVGLengthValue& rx() const { return m_rx->currentValue(); }
    const SVGLengthValue& ry() const { return m_ry->currentValue(); }

    SVGAnimatedLength& cxAnimated() { return m_cx; }
    SVGAnimatedLength& cyAnimated() { return m_cy; }
    SVGAnimatedLength& rxAnimated() { return m_rx; }
    SVGAnimatedLength& ryAnimated() { return m_ry; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGEllipseElement, SVGGeometryElement>;

private:
    SVGEllipseElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool selfHasRelativeLengths() const final { return true; }

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    Ref<SVGAnimatedLength> m_cx { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_cy { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_rx { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_ry { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
};

} // namespace WebCore
