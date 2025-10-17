/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 31, 2025.
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
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGStopElement final : public SVGElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGStopElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGStopElement);
public:
    static Ref<SVGStopElement> create(const QualifiedName&, Document&);

    Color stopColorIncludingOpacity() const;

    float offset() const { return m_offset->currentValue(); }
    SVGAnimatedNumber& offsetAnimated() { return m_offset; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGStopElement, SVGElement>;

private:
    SVGStopElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    bool isGradientStop() const final { return true; }

    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool rendererIsNeeded(const RenderStyle&) final;

    Ref<SVGAnimatedNumber> m_offset { SVGAnimatedNumber::create(this, 0) };
};

} // namespace WebCore
