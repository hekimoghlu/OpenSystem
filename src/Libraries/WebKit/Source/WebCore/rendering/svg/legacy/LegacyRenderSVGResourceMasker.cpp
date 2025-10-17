/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "LegacyRenderSVGResourceMasker.h"

#include "Element.h"
#include "ElementChildIteratorInlines.h"
#include "FloatPoint.h"
#include "Image.h"
#include "IntRect.h"
#include "LegacyRenderSVGResourceMaskerInlines.h"
#include "SVGRenderStyle.h"
#include "SVGRenderingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGResourceMasker);

LegacyRenderSVGResourceMasker::LegacyRenderSVGResourceMasker(SVGMaskElement& element, RenderStyle&& style)
    : LegacyRenderSVGResourceContainer(Type::LegacySVGResourceMasker, element, WTFMove(style))
{
}

LegacyRenderSVGResourceMasker::~LegacyRenderSVGResourceMasker() = default;

void LegacyRenderSVGResourceMasker::removeAllClientsFromCache()
{
    m_maskContentBoundaries.fill(FloatRect { });
    m_masker.clear();
}

void LegacyRenderSVGResourceMasker::removeClientFromCache(RenderElement& client)
{
    m_masker.remove(client);
}

auto LegacyRenderSVGResourceMasker::applyResource(RenderElement& renderer, const RenderStyle&, GraphicsContext*& context, OptionSet<RenderSVGResourceMode> resourceMode) -> OptionSet<ApplyResult>
{
    ASSERT(context);
    ASSERT_UNUSED(resourceMode, !resourceMode);

    bool missingMaskerData = !m_masker.contains(renderer);
    if (missingMaskerData)
        m_masker.set(renderer, makeUnique<MaskerData>());

    MaskerData* maskerData = m_masker.get(renderer);
    AffineTransform absoluteTransform = SVGRenderingContext::calculateTransformationToOutermostCoordinateSystem(renderer);
    // FIXME: This needs to be bounding box and should not use repaint rect.
    // https://bugs.webkit.org/show_bug.cgi?id=278551
    FloatRect repaintRect = renderer.repaintRectInLocalCoordinates(RepaintRectCalculation::Accurate);

    // Ignore 2D rotation, as it doesn't affect the size of the mask.
    FloatSize scale(absoluteTransform.xScale(), absoluteTransform.yScale());

    // Determine scale factor for the mask. The size of intermediate ImageBuffers shouldn't be bigger than kMaxFilterSize.
    ImageBuffer::sizeNeedsClamping(repaintRect.size(), scale);

    if (!maskerData->maskImage && !repaintRect.isEmpty()) {
        auto maskColorSpace = DestinationColorSpace::SRGB();
        auto drawColorSpace = DestinationColorSpace::SRGB();

        if (style().svgStyle().colorInterpolation() == ColorInterpolation::LinearRGB) {
#if USE(CG) || USE(SKIA)
            maskColorSpace = DestinationColorSpace::LinearSRGB();
#endif
            drawColorSpace = DestinationColorSpace::LinearSRGB();
        }
        // FIXME (149470): This image buffer should not be unconditionally unaccelerated. Making it match the context breaks alpha masking, though.
        maskerData->maskImage = context->createScaledImageBuffer(repaintRect, scale, maskColorSpace, RenderingMode::Unaccelerated);
        if (!maskerData->maskImage)
            return { };

        if (!drawContentIntoMaskImage(maskerData, drawColorSpace, &renderer))
            maskerData->maskImage = nullptr;
    }

    if (!maskerData->maskImage)
        return { };

    SVGRenderingContext::clipToImageBuffer(*context, repaintRect, scale, maskerData->maskImage, missingMaskerData);
    return { ApplyResult::ResourceApplied };
}

bool LegacyRenderSVGResourceMasker::drawContentIntoMaskImage(MaskerData* maskerData, const DestinationColorSpace& colorSpace, RenderObject* object)
{
    auto& maskImageContext = maskerData->maskImage->context();
    auto objectBoundingBox = object->objectBoundingBox();

    if (!drawContentIntoContext(maskImageContext, objectBoundingBox))
        return false;

#if !USE(CG) && !USE(SKIA)
    maskerData->maskImage->transformToColorSpace(colorSpace);
#else
    UNUSED_PARAM(colorSpace);
#endif

    // Create the luminance mask.
    if (style().svgStyle().maskType() == MaskType::Luminance)
        maskerData->maskImage->convertToLuminanceMask();

    return true;
}

bool LegacyRenderSVGResourceMasker::drawContentIntoContext(GraphicsContext& context, const FloatRect& objectBoundingBox)
{
    // Eventually adjust the mask image context according to the target objectBoundingBox.
    AffineTransform maskContentTransformation;

    if (maskElement().maskContentUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) {
        maskContentTransformation.translate(objectBoundingBox.location());
        maskContentTransformation.scale(objectBoundingBox.size());
        context.concatCTM(maskContentTransformation);
    }

    // Draw the content into the ImageBuffer.
    for (auto& child : childrenOfType<SVGElement>(protectedMaskElement())) {
        auto renderer = child.renderer();
        if (!renderer)
            continue;
        if (renderer->needsLayout())
            return false;
        const RenderStyle& style = renderer->style();
        if (style.display() == DisplayType::None || style.usedVisibility() != Visibility::Visible)
            continue;
        SVGRenderingContext::renderSubtreeToContext(context, *renderer, maskContentTransformation);
    }

    return true;
}

bool LegacyRenderSVGResourceMasker::drawContentIntoContext(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    GraphicsContextStateSaver stateSaver(context);

    context.setCompositeOperation(options.compositeOperator(), options.blendMode());

    context.translate(destinationRect.location());

    if (destinationRect.size() != sourceRect.size())
        context.scale(destinationRect.size() / sourceRect.size());

    context.translate(-sourceRect.location());

    return drawContentIntoContext(context, { { }, destinationRect.size() });
}

void LegacyRenderSVGResourceMasker::calculateMaskContentRepaintRect(RepaintRectCalculation repaintRectCalculation)
{
    for (Node* childNode = maskElement().firstChild(); childNode; childNode = childNode->nextSibling()) {
        RenderObject* renderer = childNode->renderer();
        if (!childNode->isSVGElement() || !renderer)
            continue;
        const RenderStyle& style = renderer->style();
        if (style.display() == DisplayType::None || style.usedVisibility() != Visibility::Visible)
             continue;
        m_maskContentBoundaries[repaintRectCalculation].unite(renderer->localToParentTransform().mapRect(renderer->repaintRectInLocalCoordinates(repaintRectCalculation)));
    }
}

FloatRect LegacyRenderSVGResourceMasker::resourceBoundingBox(const RenderObject& object, RepaintRectCalculation repaintRectCalculation)
{
    FloatRect objectBoundingBox = object.objectBoundingBox();
    Ref maskElement = this->maskElement();
    FloatRect maskBoundaries = SVGLengthContext::resolveRectangle<SVGMaskElement>(maskElement.ptr(), maskElement->maskUnits(), objectBoundingBox);

    // Resource was not layouted yet. Give back clipping rect of the mask.
    if (selfNeedsLayout())
        return maskBoundaries;

    if (m_maskContentBoundaries[repaintRectCalculation].isEmpty())
        calculateMaskContentRepaintRect(repaintRectCalculation);

    FloatRect maskRect = m_maskContentBoundaries[repaintRectCalculation];
    if (maskElement->maskContentUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) {
        AffineTransform transform;
        transform.translate(objectBoundingBox.location());
        transform.scale(objectBoundingBox.size());
        maskRect = transform.mapRect(maskRect);
    }

    maskRect.intersect(maskBoundaries);
    return maskRect;
}

} // namespace WebCore
