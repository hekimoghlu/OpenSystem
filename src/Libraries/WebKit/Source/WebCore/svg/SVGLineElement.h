/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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

class SVGLineElement final : public SVGGeometryElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGLineElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGLineElement);
public:
    static Ref<SVGLineElement> create(const QualifiedName&, Document&);

    const SVGLengthValue& x1() const { return m_x1->currentValue(); }
    const SVGLengthValue& y1() const { return m_y1->currentValue(); }
    const SVGLengthValue& x2() const { return m_x2->currentValue(); }
    const SVGLengthValue& y2() const { return m_y2->currentValue(); }

    SVGAnimatedLength& x1Animated() { return m_x1; }
    SVGAnimatedLength& y1Animated() { return m_y1; }
    SVGAnimatedLength& x2Animated() { return m_x2; }
    SVGAnimatedLength& y2Animated() { return m_y2; }

private:
    SVGLineElement(const QualifiedName&, Document&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGLineElement, SVGGeometryElement>;

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool supportsMarkers() const final { return true; }
    bool selfHasRelativeLengths() const final;

    Ref<SVGAnimatedLength> m_x1 { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y1 { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_x2 { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y2 { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
};

} // namespace WebCore
