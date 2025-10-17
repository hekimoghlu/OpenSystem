/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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

#include "SVGTextContentElement.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGTextPositioningElement : public SVGTextContentElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGTextPositioningElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGTextPositioningElement);
public:
    static SVGTextPositioningElement* elementFromRenderer(RenderBoxModelObject&);

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGTextPositioningElement, SVGTextContentElement>;

    const SVGLengthList& x() const { return m_x->currentValue(); }
    const SVGLengthList& y() const { return m_y->currentValue(); }
    const SVGLengthList& dx() const { return m_dx->currentValue(); }
    const SVGLengthList& dy() const { return m_dy->currentValue(); }
    const SVGNumberList& rotate() const { return m_rotate->currentValue(); }

    SVGAnimatedLengthList& xAnimated() { return m_x; }
    SVGAnimatedLengthList& yAnimated() { return m_y; }
    SVGAnimatedLengthList& dxAnimated() { return m_dx; }
    SVGAnimatedLengthList& dyAnimated() { return m_dy; }
    SVGAnimatedNumberList& rotateAnimated() { return m_rotate; }

protected:
    SVGTextPositioningElement(const QualifiedName&, Document&, UniqueRef<SVGPropertyRegistry>&&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

private:
    bool hasPresentationalHintsForAttribute(const QualifiedName&) const final;
    void collectPresentationalHintsForAttribute(const QualifiedName&, const AtomString&, MutableStyleProperties&) final;

    Ref<SVGAnimatedLengthList> m_x { SVGAnimatedLengthList::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLengthList> m_y { SVGAnimatedLengthList::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLengthList> m_dx { SVGAnimatedLengthList::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLengthList> m_dy { SVGAnimatedLengthList::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedNumberList> m_rotate { SVGAnimatedNumberList::create(this) };
};

} // namespace WebCore
