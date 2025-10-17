/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#include "LegacyRenderSVGImage.h"

#include "FloatQuad.h"
#include "GraphicsContext.h"
#include "HitTestResult.h"
#include "LayoutRepainter.h"
#include "LegacyRenderSVGResource.h"
#include "PointerEventsHitRules.h"
#include "RenderImageResource.h"
#include "RenderLayer.h"
#include "SVGElementTypeHelpers.h"
#include "SVGImageElement.h"
#include "SVGRenderStyle.h"
#include "SVGRenderingContext.h"
#include "SVGResources.h"
#include "SVGResourcesCache.h"
#include "SVGVisitedRendererTracking.h"
#include <wtf/StackStats.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRenderSVGImage);

LegacyRenderSVGImage::LegacyRenderSVGImage(SVGImageElement& element, RenderStyle&& style)
    : LegacyRenderSVGModelObject(Type::LegacySVGImage, element, WTFMove(style), SVGModelObjectFlag::UsesBoundaryCaching)
    , m_needsBoundariesUpdate(true)
    , m_needsTransformUpdate(true)
    , m_imageResource(makeUnique<RenderImageResource>())
{
    imageResource().initialize(*this);
    ASSERT(isLegacyRenderSVGImage());
}

LegacyRenderSVGImage::~LegacyRenderSVGImage() = default;

CheckedRef<RenderImageResource> LegacyRenderSVGImage::checkedImageResource() const
{
    return *m_imageResource;
}

void LegacyRenderSVGImage::willBeDestroyed()
{
    imageResource().shutdown();
    LegacyRenderSVGModelObject::willBeDestroyed();
}

SVGImageElement& LegacyRenderSVGImage::imageElement() const
{
    return downcast<SVGImageElement>(LegacyRenderSVGModelObject::element());
}

FloatRect LegacyRenderSVGImage::calculateObjectBoundingBox() const
{
    LayoutSize intrinsicSize;
    if (CachedImage* cachedImage = imageResource().cachedImage())
        intrinsicSize = cachedImage->imageSizeForRenderer(nullptr, style().usedZoom());

    Ref imageElement = this->imageElement();
    SVGLengthContext lengthContext(imageElement.ptr());

    Length width = style().width();
    Length height = style().height();

    float concreteWidth;
    if (!width.isAuto())
        concreteWidth = lengthContext.valueForLength(width, SVGLengthMode::Width);
    else if (!height.isAuto() && !intrinsicSize.isEmpty())
        concreteWidth = lengthContext.valueForLength(height, SVGLengthMode::Height) * intrinsicSize.width() / intrinsicSize.height();
    else
        concreteWidth = intrinsicSize.width();

    float concreteHeight;
    if (!height.isAuto())
        concreteHeight = lengthContext.valueForLength(height, SVGLengthMode::Height);
    else if (!width.isAuto() && !intrinsicSize.isEmpty())
        concreteHeight = lengthContext.valueForLength(width, SVGLengthMode::Width) * intrinsicSize.height() / intrinsicSize.width();
    else
        concreteHeight = intrinsicSize.height();

    return { imageElement->x().value(lengthContext), imageElement->y().value(lengthContext), concreteWidth, concreteHeight };
}

bool LegacyRenderSVGImage::updateImageViewport()
{
    FloatRect oldBoundaries = m_objectBoundingBox;
    m_objectBoundingBox = calculateObjectBoundingBox();

    bool updatedViewport = false;
    URL imageSourceURL = document().completeURL(imageElement().imageSourceURL());

    // Images with preserveAspectRatio=none should force non-uniform scaling. This can be achieved
    // by setting the image's container size to its intrinsic size.
    // See: http://www.w3.org/TR/SVG/single-page.html, 7.8 The â€˜preserveAspectRatioâ€™ attribute.
    if (imageElement().preserveAspectRatio().align() == SVGPreserveAspectRatioValue::SVG_PRESERVEASPECTRATIO_NONE) {
        if (CachedImage* cachedImage = imageResource().cachedImage()) {
            LayoutSize intrinsicSize = cachedImage->imageSizeForRenderer(nullptr, style().usedZoom());
            if (intrinsicSize != imageResource().imageSize(style().usedZoom())) {
                imageResource().setContainerContext(roundedIntSize(intrinsicSize), imageSourceURL);
                updatedViewport = true;
            }
        }
    }

    if (oldBoundaries != m_objectBoundingBox) {
        if (!updatedViewport)
            imageResource().setContainerContext(enclosingIntRect(m_objectBoundingBox).size(), imageSourceURL);
        updatedViewport = true;
        m_needsBoundariesUpdate = true;
    }

    return updatedViewport;
}

void LegacyRenderSVGImage::layout()
{
    StackStats::LayoutCheckPoint layoutCheckPoint;
    ASSERT(needsLayout());

    auto checkForRepaintOverride = !selfNeedsLayout() ? LayoutRepainter::CheckForRepaint::No : SVGRenderSupport::checkForSVGRepaintDuringLayout(*this);
    LayoutRepainter repainter(*this, checkForRepaintOverride, { }, RepaintOutlineBounds::No);
    updateImageViewport();

    bool transformOrBoundariesUpdate = m_needsTransformUpdate || m_needsBoundariesUpdate;
    if (m_needsTransformUpdate) {
        m_localTransform = imageElement().animatedLocalTransform();
        m_needsTransformUpdate = false;
    }

    if (m_needsBoundariesUpdate) {
        m_repaintBoundingBox = m_objectBoundingBox;
        SVGRenderSupport::intersectRepaintRectWithResources(*this, m_repaintBoundingBox);
        m_needsBoundariesUpdate = false;
    }

    // Invalidate all resources of this client if our layout changed.
    if (everHadLayout() && selfNeedsLayout())
        SVGResourcesCache::clientLayoutChanged(*this);

    // If our bounds changed, notify the parents.
    if (transformOrBoundariesUpdate) {
        if (CheckedPtr parent = this->parent())
            parent->invalidateCachedBoundaries();
    }

    repainter.repaintAfterLayout();
    clearNeedsLayout();
}

void LegacyRenderSVGImage::paint(PaintInfo& paintInfo, const LayoutPoint&)
{
    if (paintInfo.context().paintingDisabled() || paintInfo.phase != PaintPhase::Foreground
        || style().usedVisibility() == Visibility::Hidden || !imageResource().cachedImage())
        return;

    FloatRect boundingBox = repaintRectInLocalCoordinates();
    if (!SVGRenderSupport::paintInfoIntersectsRepaintRect(boundingBox, m_localTransform, paintInfo))
        return;

    PaintInfo childPaintInfo(paintInfo);
    GraphicsContextStateSaver stateSaver(childPaintInfo.context());
    childPaintInfo.applyTransform(m_localTransform);

    if (childPaintInfo.phase == PaintPhase::Foreground) {
        SVGRenderingContext renderingContext(*this, childPaintInfo);

        if (renderingContext.isRenderingPrepared()) {
            if (style().svgStyle().bufferedRendering() == BufferedRendering::Static && renderingContext.bufferForeground(m_bufferedForeground))
                return;

            paintForeground(childPaintInfo);
        }
    }

    if (style().outlineWidth())
        paintOutline(childPaintInfo, IntRect(boundingBox));
}

void LegacyRenderSVGImage::paintForeground(PaintInfo& paintInfo)
{
    RefPtr<Image> image = imageResource().image();
    if (!image)
        return;

    FloatRect destRect = m_objectBoundingBox;
    FloatRect srcRect(0, 0, image->width(), image->height());

    imageElement().preserveAspectRatio().transformRect(destRect, srcRect);

    paintInfo.context().drawImage(*image, destRect, srcRect);
}

void LegacyRenderSVGImage::invalidateBufferedForeground()
{
    m_bufferedForeground = nullptr;
}

bool LegacyRenderSVGImage::nodeAtFloatPoint(const HitTestRequest& request, HitTestResult& result, const FloatPoint& pointInParent, HitTestAction hitTestAction)
{
    // We only draw in the forground phase, so we only hit-test then.
    if (hitTestAction != HitTestForeground)
        return false;

    PointerEventsHitRules hitRules(PointerEventsHitRules::HitTestingTargetType::SVGImage, request, usedPointerEvents());
    if (isVisibleToHitTesting(style(), request) || !hitRules.requireVisible) {
        static NeverDestroyed<SVGVisitedRendererTracking::VisitedSet> s_visitedSet;

        SVGVisitedRendererTracking recursionTracking(s_visitedSet);
        if (recursionTracking.isVisiting(*this))
            return false;

        SVGVisitedRendererTracking::Scope recursionScope(recursionTracking, *this);

        FloatPoint localPoint = valueOrDefault(localToParentTransform().inverse()).mapPoint(pointInParent);
        if (!SVGRenderSupport::pointInClippingArea(*this, localPoint))
            return false;

        if (hitRules.canHitFill) {
            if (m_objectBoundingBox.contains(localPoint)) {
                updateHitTestResult(result, LayoutPoint(localPoint));
                if (result.addNodeToListBasedTestResult(protectedNodeForHitTest().get(), request, flooredLayoutPoint(localPoint)) == HitTestProgress::Stop)
                    return true;
            }
        }
    }

    return false;
}

void LegacyRenderSVGImage::imageChanged(WrappedImagePtr, const IntRect*)
{
    if (!parent())
        return;
    // The image resource defaults to nullImage until the resource arrives.
    // This empty image may be cached by SVG resources which must be invalidated.
    if (auto* resources = SVGResourcesCache::cachedResourcesForRenderer(*this))
        resources->removeClientFromCacheAndMarkForInvalidation(*this);

    // Eventually notify parent resources, that we've changed.
    LegacyRenderSVGResource::markForLayoutAndParentResourceInvalidation(*this, false);

    // Update the SVGImageCache sizeAndScales entry in case image loading finished after layout.
    // (https://bugs.webkit.org/show_bug.cgi?id=99489)
    m_objectBoundingBox = FloatRect();
    if (updateImageViewport())
        setNeedsLayout();

    invalidateBufferedForeground();

    repaint();
}

void LegacyRenderSVGImage::addFocusRingRects(Vector<LayoutRect>& rects, const LayoutPoint&, const RenderLayerModelObject*) const
{
    // this is called from paint() after the localTransform has already been applied
    LayoutRect contentRect = LayoutRect(repaintRectInLocalCoordinates());
    if (!contentRect.isEmpty())
        rects.append(contentRect);
}

} // namespace WebCore
