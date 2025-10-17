/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 7, 2023.
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
#include "SVGFEImageElement.h"

#include "CachedImage.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "Document.h"
#include "FEImage.h"
#include "Image.h"
#include "LegacyRenderSVGResource.h"
#include "RenderObject.h"
#include "SVGElementInlines.h"
#include "SVGNames.h"
#include "SVGPreserveAspectRatioValue.h"
#include "SVGRenderingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGFEImageElement);

inline SVGFEImageElement::SVGFEImageElement(const QualifiedName& tagName, Document& document)
    : SVGFilterPrimitiveStandardAttributes(tagName, document, makeUniqueRef<PropertyRegistry>(*this))
    , SVGURIReference(this)
{
    ASSERT(hasTagName(SVGNames::feImageTag));

    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        PropertyRegistry::registerProperty<SVGNames::preserveAspectRatioAttr, &SVGFEImageElement::m_preserveAspectRatio>();
    });
}

Ref<SVGFEImageElement> SVGFEImageElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGFEImageElement(tagName, document));
}

SVGFEImageElement::~SVGFEImageElement()
{
    clearResourceReferences();
}

bool SVGFEImageElement::renderingTaintsOrigin() const
{
    if (!m_cachedImage)
        return false;
    RefPtr image = m_cachedImage->image();
    return image && image->renderingTaintsOrigin();
}

void SVGFEImageElement::clearResourceReferences()
{
    if (CachedResourceHandle cachedImage = std::exchange(m_cachedImage, nullptr))
        cachedImage->removeClient(*this);

    removeElementReference();
}

void SVGFEImageElement::requestImageResource()
{
    ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
    options.contentSecurityPolicyImposition = isInUserAgentShadowTree() ? ContentSecurityPolicyImposition::SkipPolicyCheck : ContentSecurityPolicyImposition::DoPolicyCheck;

    CachedResourceRequest request(ResourceRequest(document().completeURL(href())), options);
    request.setInitiator(*this);
    m_cachedImage = document().protectedCachedResourceLoader()->requestImage(WTFMove(request)).value_or(nullptr);

    if (CachedResourceHandle cachedImage = m_cachedImage)
        cachedImage->addClient(*this);
}

void SVGFEImageElement::buildPendingResource()
{
    clearResourceReferences();
    if (!isConnected())
        return;

    auto target = SVGURIReference::targetElementFromIRIString(href(), treeScopeForSVGReferences());
    if (!target.element) {
        if (target.identifier.isEmpty())
            requestImageResource();
        else {
            treeScopeForSVGReferences().addPendingSVGResource(target.identifier, *this);
            ASSERT(hasPendingResources());
        }
    } else if (RefPtr element = dynamicDowncast<SVGElement>(*target.element))
        element->addReferencingElement(*this);

    updateSVGRendererForElementChange();
}

void SVGFEImageElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::preserveAspectRatioAttr)
        m_preserveAspectRatio->setBaseValInternal(SVGPreserveAspectRatioValue { newValue });

    SVGURIReference::parseAttribute(name, newValue);
    SVGFilterPrimitiveStandardAttributes::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}

void SVGFEImageElement::svgAttributeChanged(const QualifiedName& attrName)
{
    if (PropertyRegistry::isKnownAttribute(attrName)) {
        ASSERT(attrName == SVGNames::preserveAspectRatioAttr);
        InstanceInvalidationGuard guard(*this);
        updateSVGRendererForElementChange();
        return;
    }

    if (SVGURIReference::isKnownAttribute(attrName)) {
        InstanceInvalidationGuard guard(*this);
        buildPendingResource();
        markFilterEffectForRebuild();
        return;
    }

    SVGFilterPrimitiveStandardAttributes::svgAttributeChanged(attrName);
}

Node::InsertedIntoAncestorResult SVGFEImageElement::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    SVGFilterPrimitiveStandardAttributes::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (!insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::Done;
    return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
}

void SVGFEImageElement::didFinishInsertingNode()
{
    SVGFilterPrimitiveStandardAttributes::didFinishInsertingNode();
    buildPendingResource();
}

void SVGFEImageElement::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    SVGFilterPrimitiveStandardAttributes::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (removalType.disconnectedFromDocument)
        clearResourceReferences();
}

void SVGFEImageElement::notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess)
{
    if (!isConnected())
        return;

    RefPtr parent = parentElement();

    if (!parent || !parent->hasTagName(SVGNames::filterTag))
        return;

    CheckedPtr parentRenderer = parent->renderer();
    if (!parentRenderer)
        return;

    // FIXME: [LBSE] Implement filters.
    if (document().settings().layerBasedSVGEngineEnabled())
        return;

    LegacyRenderSVGResource::markForLayoutAndParentResourceInvalidation(*parentRenderer);
}

std::tuple<RefPtr<ImageBuffer>, FloatRect> SVGFEImageElement::imageBufferForEffect(const GraphicsContext& destinationContext) const
{
    auto target = SVGURIReference::targetElementFromIRIString(href(), const_cast<SVGFEImageElement&>(*this).treeScopeForSVGReferences());
    if (!is<SVGElement>(target.element))
        return { };

    if (isDescendantOrShadowDescendantOf(target.element.get()))
        return { };

    RefPtr contextNode = static_pointer_cast<SVGElement>(target.element);
    CheckedPtr renderer = contextNode->renderer();
    if (!renderer)
        return { };

    auto absoluteTransform = SVGRenderingContext::calculateTransformationToOutermostCoordinateSystem(*renderer);
    if (!absoluteTransform.isInvertible())
        return { };

    // Ignore 2D rotation, as it doesn't affect the image size.
    FloatSize scale(absoluteTransform.xScale(), absoluteTransform.yScale());
    auto imageRect = renderer->repaintRectInLocalCoordinates();

    RefPtr imageBuffer = destinationContext.createScaledImageBuffer(imageRect, scale);
    if (!imageBuffer)
        return { };

    auto& context = imageBuffer->context();
    SVGRenderingContext::renderSubtreeToContext(context, *renderer, AffineTransform());

    return { WTFMove(imageBuffer), imageRect };
}

RefPtr<FilterEffect> SVGFEImageElement::createFilterEffect(const FilterEffectVector&, const GraphicsContext& destinationContext) const
{
    if (CachedResourceHandle cachedImage = m_cachedImage) {
        RefPtr image = cachedImage->imageForRenderer(renderer());
        if (!image || image->isNull())
            return nullptr;

        RefPtr nativeImage = image->currentPreTransformedNativeImage();
        if (!nativeImage)
            return nullptr;

        auto imageRect = FloatRect { { }, image->size() };
        return FEImage::create({ nativeImage.releaseNonNull() }, imageRect, preserveAspectRatio());
    }

    auto [imageBuffer, imageRect] = imageBufferForEffect(destinationContext);
    if (!imageBuffer)
        return nullptr;

    return FEImage::create({ imageBuffer.releaseNonNull() }, imageRect, preserveAspectRatio());
}

void SVGFEImageElement::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    SVGFilterPrimitiveStandardAttributes::addSubresourceAttributeURLs(urls);

    addSubresourceURL(urls, document().completeURL(href()));
}

} // namespace WebCore
