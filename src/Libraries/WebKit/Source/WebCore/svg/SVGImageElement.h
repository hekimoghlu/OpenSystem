/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#include "SVGImageLoader.h"
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SVGImageElement final : public SVGGraphicsElement, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGImageElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGImageElement);
public:
    static Ref<SVGImageElement> create(const QualifiedName&, Document&);

    CachedImage* cachedImage() const;
    bool renderingTaintsOrigin() const;
    const AtomString& imageSourceURL() const final;

    const SVGLengthValue& x() const { return m_x->currentValue(); }
    const SVGLengthValue& y() const { return m_y->currentValue(); }
    const SVGLengthValue& width() const { return m_width->currentValue(); }
    const SVGLengthValue& height() const { return m_height->currentValue(); }
    const SVGPreserveAspectRatioValue& preserveAspectRatio() const { return m_preserveAspectRatio->currentValue(); }

    void setCrossOrigin(const AtomString&);
    String crossOrigin() const;

    SVGAnimatedLength& xAnimated() { return m_x; }
    SVGAnimatedLength& yAnimated() { return m_y; }
    SVGAnimatedLength& widthAnimated() { return m_width; }
    SVGAnimatedLength& heightAnimated() { return m_height; }
    SVGAnimatedPreserveAspectRatio& preserveAspectRatioAnimated() { return m_preserveAspectRatio; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGImageElement, SVGGraphicsElement, SVGURIReference>;

    void decode(Ref<DeferredPromise>&&);

private:
    SVGImageElement(const QualifiedName&, Document&);
    
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    void didAttachRenderers() final;
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;

    void addSubresourceAttributeURLs(ListHashSet<URL>&) const final;
    bool haveLoadedRequiredResources() final;

    bool isValid() const final { return SVGTests::isValid(); }
    bool selfHasRelativeLengths() const final { return true; }
    void didMoveToNewDocument(Document& oldDocument, Document& newDocument) final;

    Ref<SVGAnimatedLength> m_x { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_y { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedLength> m_width { SVGAnimatedLength::create(this, SVGLengthMode::Width) };
    Ref<SVGAnimatedLength> m_height { SVGAnimatedLength::create(this, SVGLengthMode::Height) };
    Ref<SVGAnimatedPreserveAspectRatio> m_preserveAspectRatio { SVGAnimatedPreserveAspectRatio::create(this) };
    SVGImageLoader m_imageLoader;
};

} // namespace WebCore
