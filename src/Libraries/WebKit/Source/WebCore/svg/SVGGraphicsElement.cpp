/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 23, 2022.
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
#include "config.h"
#include "SVGGraphicsElement.h"

#include "LegacyRenderSVGPath.h"
#include "LegacyRenderSVGResource.h"
#include "RenderAncestorIterator.h"
#include "RenderElementInlines.h"
#include "RenderLayer.h"
#include "RenderLayerInlines.h"
#include "RenderSVGHiddenContainer.h"
#include "RenderSVGPath.h"
#include "RenderSVGResourceMasker.h"
#include "RenderSVGResourcePattern.h"
#include "SVGMatrix.h"
#include "SVGNames.h"
#include "SVGPathData.h"
#include "SVGRect.h"
#include "SVGRenderSupport.h"
#include "SVGSVGElement.h"
#include "SVGStringList.h"
#include "TransformOperationData.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGGraphicsElement);

SVGGraphicsElement::SVGGraphicsElement(const QualifiedName& tagName, Document& document, UniqueRef<SVGPropertyRegistry>&& propertyRegistry, OptionSet<TypeFlag> typeFlags)
    : SVGElement(tagName, document, WTFMove(propertyRegistry), typeFlags)
    , SVGTests(this)
    , m_shouldIsolateBlending(false)
{
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::transformAttr, &SVGGraphicsElement::m_transform>();
    });
}

SVGGraphicsElement::~SVGGraphicsElement() = default;

Ref<SVGMatrix> SVGGraphicsElement::getCTMForBindings()
{
    return SVGMatrix::create(getCTM());
}

AffineTransform SVGGraphicsElement::getCTM(StyleUpdateStrategy styleUpdateStrategy)
{
    return SVGLocatable::computeCTM(this, SVGLocatable::NearestViewportScope, styleUpdateStrategy);
}

Ref<SVGMatrix> SVGGraphicsElement::getScreenCTMForBindings()
{
    return SVGMatrix::create(getScreenCTM());
}

AffineTransform SVGGraphicsElement::getScreenCTM(StyleUpdateStrategy styleUpdateStrategy)
{
    return SVGLocatable::computeCTM(this, SVGLocatable::ScreenScope, styleUpdateStrategy);
}

Ref<const SVGTransformList> SVGGraphicsElement::protectedTransform() const
{
    return m_transform->currentValue();
}

AffineTransform SVGGraphicsElement::animatedLocalTransform() const
{
    // LBSE handles transforms via RenderLayer, no need to handle CSS transforms here.
    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (m_supplementalTransform)
            return *m_supplementalTransform * transform().concatenate();
        return protectedTransform()->concatenate();
    }

    AffineTransform matrix;

    CheckedPtr renderer = this->renderer();
    CheckedPtr style = renderer ? &renderer->style() : nullptr;
    bool hasSpecifiedTransform = style && style->hasTransform();

    // Honor any of the transform-related CSS properties if set.
    if (hasSpecifiedTransform || (style && (style->translate() || style->scale() || style->rotate()))) {
        // Note: objectBoundingBox is an emptyRect for elements like pattern or clipPath.
        // See the "Object bounding box units" section of http://dev.w3.org/csswg/css3-transforms/
        TransformationMatrix transform;
        style->applyTransform(transform, TransformOperationData(renderer->transformReferenceBoxRect(), renderer.get()));

        // Flatten any 3D transform.
        matrix = transform.toAffineTransform();
    }

    // If we didn't have the CSS "transform" property set, we must account for the "transform" attribute.
    if (!hasSpecifiedTransform && style && !transform().isEmpty()) {
        auto t = style->computeTransformOrigin(renderer->transformReferenceBoxRect()).xy();
        matrix.translate(t);
        matrix *= transform().concatenate();
        matrix.translate(-t.x(), -t.y());
    }

    if (m_supplementalTransform)
        return *m_supplementalTransform * matrix;
    return matrix;
}

AffineTransform* SVGGraphicsElement::ensureSupplementalTransform()
{
    if (!m_supplementalTransform)
        m_supplementalTransform = makeUnique<AffineTransform>();
    return m_supplementalTransform.get();
}

void SVGGraphicsElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::transformAttr)
        Ref { m_transform }->baseVal()->parse(newValue);

    SVGTests::parseAttribute(name, newValue);
    SVGElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGGraphicsElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::transformAttr);
        InstanceInvalidationGuard guard(*this);

        if (document().settings().layerBasedSVGEngineEnabled()) {
            if (CheckedPtr layerRenderer = dynamicDowncast<RenderLayerModelObject>(renderer()))
                layerRenderer->repaintOrRelayoutAfterSVGTransformChange();
            return;
        }

        if (CheckedPtr renderer = this->renderer())
            renderer->setNeedsTransformUpdate();
        updateSVGRendererForElementChange();
        return;
    }

    SVGElement::svgAttributeChanged(attrName);
    SVGTests::svgAttributeChanged(attrName);
}

SVGElement* SVGGraphicsElement::nearestViewportElement() const
{
    return SVGTransformable::nearestViewportElement(this);
}

SVGElement* SVGGraphicsElement::farthestViewportElement() const
{
    return SVGTransformable::farthestViewportElement(this);
}

Ref<SVGRect> SVGGraphicsElement::getBBoxForBindings()
{
    return SVGRect::create(getBBox());
}

FloatRect SVGGraphicsElement::getBBox(StyleUpdateStrategy styleUpdateStrategy)
{
    return SVGTransformable::getBBox(this, styleUpdateStrategy);
}

RenderPtr<RenderElement> SVGGraphicsElement::createElementRenderer(RenderStyle&& style, const RenderTreePosition&)
{
    if (document().settings().layerBasedSVGEngineEnabled())
        return createRenderer<RenderSVGPath>(*this, WTFMove(style));
    return createRenderer<LegacyRenderSVGPath>(*this, WTFMove(style));
}

void SVGGraphicsElement::didAttachRenderers()
{
    if (document().settings().layerBasedSVGEngineEnabled()) {
        if (CheckedPtr svgRenderer = dynamicDowncast<RenderLayerModelObject>(renderer()); svgRenderer && lineageOfType<RenderSVGHiddenContainer>(*svgRenderer).first()) {
            if (CheckedPtr layer = svgRenderer->layer())
                layer->dirtyVisibleContentStatus();
        }
    }
}

Path SVGGraphicsElement::toClipPath()
{
    RELEASE_ASSERT(!document().settings().layerBasedSVGEngineEnabled());

    Path path = pathFromGraphicsElement(*this);
    // FIXME: How do we know the element has done a layout?
    path.transform(animatedLocalTransform());
    return path;
}

void SVGGraphicsElement::invalidateResourceImageBuffersIfNeeded()
{
    if (!document().settings().layerBasedSVGEngineEnabled())
        return;
    if (CheckedPtr svgRenderer = dynamicDowncast<RenderLayerModelObject>(renderer())) {
        if (CheckedPtr container = svgRenderer->enclosingLayer()->enclosingSVGHiddenOrResourceContainer()) {
            if (auto* maskRenderer = dynamicDowncast<RenderSVGResourceMasker>(container.get()))
                maskRenderer->invalidateMask();
            if (auto* patternRenderer = dynamicDowncast<RenderSVGResourcePattern>(container.get()))
                patternRenderer->invalidatePattern(RenderSVGResourcePattern::SuppressRepaint::Yes);
        }
    }
}

}
