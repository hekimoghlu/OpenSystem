/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#include "RenderSVGResourceMasker.h"

#include "Element.h"
#include "ElementIterator.h"
#include "FloatPoint.h"
#include "Image.h"
#include "ImageBuffer.h"
#include "IntRect.h"
#include "RenderLayer.h"
#include "RenderLayerInlines.h"
#include "RenderSVGModelObjectInlines.h"
#include "RenderSVGResourceMaskerInlines.h"
#include "SVGContainerLayout.h"
#include "SVGElementTypeHelpers.h"
#include "SVGGraphicsElement.h"
#include "SVGLengthContext.h"
#include "SVGRenderStyle.h"
#include "SVGVisitedRendererTracking.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderSVGResourceMasker);

RenderSVGResourceMasker::RenderSVGResourceMasker(SVGMaskElement& element, RenderStyle&& style)
    : RenderSVGResourceContainer(Type::SVGResourceMasker, element, WTFMove(style))
{
}

RenderSVGResourceMasker::~RenderSVGResourceMasker() = default;

static RefPtr<ImageBuffer> createImageBuffer(const FloatRect& targetRect, const AffineTransform& absoluteTransform, const DestinationColorSpace& colorSpace, const GraphicsContext* context)
{
    IntRect paintRect = enclosingIntRect(absoluteTransform.mapRect(targetRect));
    // Don't create empty ImageBuffers.
    if (paintRect.isEmpty())
        return nullptr;

    FloatSize scale;
    FloatSize clampedSize = ImageBuffer::clampedSize(paintRect.size(), scale);

    UNUSED_PARAM(context);
    auto imageBuffer = ImageBuffer::create(clampedSize, RenderingMode::Unaccelerated, RenderingPurpose::Unspecified, 1, colorSpace, ImageBufferPixelFormat::BGRA8);
    if (!imageBuffer)
        return nullptr;

    AffineTransform transform;
    transform.scale(scale).translate(-paintRect.location()).multiply(absoluteTransform);

    auto& imageContext = imageBuffer->context();
    imageContext.concatCTM(transform);

    return imageBuffer;
}

void RenderSVGResourceMasker::applyMask(PaintInfo& paintInfo, const RenderLayerModelObject& targetRenderer, const LayoutPoint& adjustedPaintOffset)
{
    ASSERT(hasLayer());
    ASSERT(layer()->isSelfPaintingLayer());
    ASSERT(targetRenderer.hasLayer());

    static NeverDestroyed<SVGVisitedRendererTracking::VisitedSet> s_visitedSet;

    SVGVisitedRendererTracking recursionTracking(s_visitedSet);
    if (recursionTracking.isVisiting(*this))
        return;

    SVGVisitedRendererTracking::Scope recursionScope(recursionTracking, *this);

    auto& context = paintInfo.context();
    GraphicsContextStateSaver stateSaver(context);

    auto objectBoundingBox = targetRenderer.objectBoundingBox();
    auto boundingBoxTopLeftCorner = flooredLayoutPoint(objectBoundingBox.minXMinYCorner());
    auto coordinateSystemOriginTranslation = adjustedPaintOffset - boundingBoxTopLeftCorner;
    if (!coordinateSystemOriginTranslation.isZero())
        context.translate(coordinateSystemOriginTranslation);

    // FIXME: This needs to be bounding box and should not use repaint rect.
    // https://bugs.webkit.org/show_bug.cgi?id=278551
    auto repaintBoundingBox = targetRenderer.repaintRectInLocalCoordinates(RepaintRectCalculation::Accurate);
    auto absoluteTransform = context.getCTM(GraphicsContext::DefinitelyIncludeDeviceScale);

    auto maskColorSpace = DestinationColorSpace::SRGB();
    auto drawColorSpace = DestinationColorSpace::SRGB();

    Ref svgStyle = style().svgStyle();
    if (svgStyle->colorInterpolation() == ColorInterpolation::LinearRGB) {
#if USE(CG) || USE(SKIA)
        maskColorSpace = DestinationColorSpace::LinearSRGB();
#endif
        drawColorSpace = DestinationColorSpace::LinearSRGB();
    }

    RefPtr<ImageBuffer> maskImage = m_masker.get(targetRenderer);
    bool missingMaskerData = !maskImage;
    if (missingMaskerData) {
        // FIXME: try to use GraphicsContext::createScaledImageBuffer instead.
        maskImage = createImageBuffer(repaintBoundingBox, absoluteTransform, maskColorSpace, &context);
        if (!maskImage)
            return;
    }

    context.setCompositeOperation(CompositeOperator::DestinationIn);
    context.beginTransparencyLayer(1);

    if (missingMaskerData) {
        drawContentIntoContext(maskImage->context(), objectBoundingBox);

#if !USE(CG) && !USE(SKIA)
        maskImage->transformToColorSpace(drawColorSpace);
#else
        UNUSED_PARAM(drawColorSpace);
#endif

        if (svgStyle->maskType() == MaskType::Luminance)
            maskImage->convertToLuminanceMask();
        m_masker.set(targetRenderer, maskImage);
    }
    context.setCompositeOperation(CompositeOperator::SourceOver);

    // The mask image has been created in the absolute coordinate space, as the image should not be scaled.
    // So the actual masking process has to be done in the absolute coordinate space as well.
    FloatRect absoluteTargetRect = enclosingIntRect(absoluteTransform.mapRect(repaintBoundingBox));
    context.concatCTM(absoluteTransform.inverse().value_or(AffineTransform()));
    context.drawImageBuffer(*maskImage, absoluteTargetRect);
    context.endTransparencyLayer();
}

FloatRect RenderSVGResourceMasker::resourceBoundingBox(const RenderObject& object, RepaintRectCalculation repaintRectCalculation)
{
    auto targetBoundingBox = object.objectBoundingBox();
    static NeverDestroyed<SVGVisitedRendererTracking::VisitedSet> s_visitedSet;

    SVGVisitedRendererTracking recursionTracking(s_visitedSet);
    if (recursionTracking.isVisiting(*this))
        return targetBoundingBox;

    SVGVisitedRendererTracking::Scope recursionScope(recursionTracking, *this);

    Ref maskElement = this->maskElement();
    auto maskRect = maskElement->calculateMaskContentRepaintRect(repaintRectCalculation);
    if (maskElement->maskContentUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) {
        AffineTransform contentTransform;
        contentTransform.translate(targetBoundingBox.location());
        contentTransform.scale(targetBoundingBox.size());
        maskRect = contentTransform.mapRect(maskRect);
    }

    auto maskBoundaries = SVGLengthContext::resolveRectangle<SVGMaskElement>(maskElement.ptr(), maskElement->maskUnits(), targetBoundingBox);
    maskRect.intersect(maskBoundaries);
    if (maskRect.isEmpty())
        return targetBoundingBox;
    return maskRect;
}

void RenderSVGResourceMasker::removeReferencingCSSClient(const RenderElement& client)
{
    if (auto renderer = dynamicDowncast<RenderLayerModelObject>(client))
        m_masker.remove(renderer);
}

bool RenderSVGResourceMasker::drawContentIntoContext(GraphicsContext& context, const FloatRect& objectBoundingBox)
{
    // Eventually adjust the mask image context according to the target objectBoundingBox.
    AffineTransform maskContentTransformation;

    if (protectedMaskElement()->maskContentUnits() == SVGUnitTypes::SVG_UNIT_TYPE_OBJECTBOUNDINGBOX) {
        maskContentTransformation.translate(objectBoundingBox.location());
        maskContentTransformation.scale(objectBoundingBox.size());
    }

    // Draw the content into the ImageBuffer.
    checkedLayer()->paintSVGResourceLayer(context, maskContentTransformation);
    return true;
}

bool RenderSVGResourceMasker::drawContentIntoContext(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    GraphicsContextStateSaver stateSaver(context);
    context.setCompositeOperation(options.compositeOperator(), options.blendMode());
    context.translate(destinationRect.location());

    if (destinationRect.size() != sourceRect.size())
        context.scale(destinationRect.size() / sourceRect.size());

    context.translate(-sourceRect.location());
    return drawContentIntoContext(context, { { }, destinationRect.size() });
}

}
