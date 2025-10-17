/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#include "SVGTransformable.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class AffineTransform;
class Path;
class SVGRect;
class SVGMatrix;

class SVGGraphicsElement : public SVGElement, public SVGTransformable, public SVGTests {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGGraphicsElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGGraphicsElement);
public:
    virtual ~SVGGraphicsElement();

    Ref<SVGMatrix> getCTMForBindings();
    AffineTransform getCTM(StyleUpdateStrategy = AllowStyleUpdate) override;

    Ref<SVGMatrix> getScreenCTMForBindings();
    AffineTransform getScreenCTM(StyleUpdateStrategy = AllowStyleUpdate) override;

    SVGElement* nearestViewportElement() const override;
    SVGElement* farthestViewportElement() const override;

    AffineTransform localCoordinateSpaceTransform(SVGLocatable::CTMScope mode) const override { return SVGTransformable::localCoordinateSpaceTransform(mode); }
    AffineTransform animatedLocalTransform() const override;
    AffineTransform* ensureSupplementalTransform() override;
    AffineTransform* supplementalTransform() const override { return m_supplementalTransform.get(); }

    virtual bool hasTransformRelatedAttributes() const { return !transform().concatenate().isIdentity() || m_supplementalTransform; }

    Ref<SVGRect> getBBoxForBindings();
    FloatRect getBBox(StyleUpdateStrategy = AllowStyleUpdate) override;

    bool shouldIsolateBlending() const { return m_shouldIsolateBlending; }
    void setShouldIsolateBlending(bool isolate) { m_shouldIsolateBlending = isolate; }

    // "base class" methods for all the elements which render as paths
    virtual Path toClipPath();
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) override;

    size_t approximateMemoryCost() const override { return sizeof(*this); }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGGraphicsElement, SVGElement, SVGTests>;

    const SVGTransformList& transform() const { return m_transform->currentValue(); }
    Ref<const SVGTransformList> protectedTransform() const;
    SVGAnimatedTransformList& transformAnimated() { return m_transform; }

protected:
    SVGGraphicsElement(const QualifiedName&, Document&, UniqueRef<SVGPropertyRegistry>&&, OptionSet<TypeFlag> = { });

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;
    void didAttachRenderers() override;

    void invalidateResourceImageBuffersIfNeeded();

private:
    bool isSVGGraphicsElement() const override { return true; }

    // Used by <animateMotion>
    std::unique_ptr<AffineTransform> m_supplementalTransform;

    // Used to isolate blend operations caused by masking.
    bool m_shouldIsolateBlending;

    Ref<SVGAnimatedTransformList> m_transform { SVGAnimatedTransformList::create(this) };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SVGGraphicsElement)
    static bool isType(const WebCore::SVGElement& element) { return element.isSVGGraphicsElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* svgElement = dynamicDowncast<WebCore::SVGElement>(node);
        return svgElement && isType(*svgElement);
    }
SPECIALIZE_TYPE_TRAITS_END()
