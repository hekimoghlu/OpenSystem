/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "SVGTests.h"
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class StyleCursorImage;

class SVGCursorElement final : public SVGElement, public SVGTests, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGCursorElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGCursorElement);
public:
    static Ref<SVGCursorElement> create(const QualifiedName&, Document&);

    virtual ~SVGCursorElement();

    void addClient(StyleCursorImage&);
    void removeClient(StyleCursorImage&);

    const SVGLengthValue& x() const { return m_x->currentValue(); }
    const SVGLengthValue& y() const { return m_y->currentValue(); }

    SVGAnimatedLength& xAnimated() { return m_x; }
    SVGAnimatedLength& yAnimated() { return m_y; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGCursorElement, SVGElement, SVGTests, SVGURIReference>;

private:
    SVGCursorElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool rendererIsNeeded(const RenderStyle&) final { return false; }

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;

    Ref<SVGAnimatedLength> m_x { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    WeakHashSet<StyleCursorImage> m_clients;
};

} // namespace WebCore
