/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

struct LinearGradientAttributes;

class SVGLinearGradientElement final : public SVGGradientElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGLinearGradientElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGLinearGradientElement);
public:
    static Ref<SVGLinearGradientElement> create(const QualifiedName&, Document&);

    bool collectGradientAttributes(LinearGradientAttributes&);

    const SVGLengthValue& x1() const { return m_x1->currentValue(); }
    const SVGLengthValue& y1() const { return m_y1->currentValue(); }
    const SVGLengthValue& x2() const { return m_x2->currentValue(); }
    const SVGLengthValue& y2() const { return m_y2->currentValue(); }

    SVGAnimatedLength& x1Animated() { return m_x1; }
    SVGAnimatedLength& y1Animated() { return m_y1; }
    SVGAnimatedLength& x2Animated() { return m_x2; }
    SVGAnimatedLength& y2Animated() { return m_y2; }

private:
    SVGLinearGradientElement(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGLinearGradientElement, SVGGradientElement>;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;

    bool selfHasRelativeLengths() const override;
    bool supportsFocus() const final { return false; }

    Ref<SVGAnimatedLength> m_x1 { SVGAnimatedLength::create(this, SVGLengthMode::Width, "0%"_s) };
    Ref<SVGAnimatedLength> m_y1 { SVGAnimatedLength::create(this, SVGLengthMode::Height, "0%"_s) };
    Ref<SVGAnimatedLength> m_x2 { SVGAnimatedLength::create(this, SVGLengthMode::Width, "100%"_s) };
    Ref<SVGAnimatedLength> m_y2 { SVGAnimatedLength::create(this, SVGLengthMode::Height, "0%"_s) };
};

} // namespace WebCore
