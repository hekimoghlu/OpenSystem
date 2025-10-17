/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "DragImage.h"

#include "BitmapImage.h"
#include "FrameSnapshotting.h"
#include "ImageBuffer.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "NotImplemented.h"
#include "Position.h"
#include "RenderElement.h"
#include "RenderSelection.h"
#include "RenderView.h"
#include "SimpleRange.h"
#include "TextIndicator.h"

namespace WebCore {

#if PLATFORM(COCOA)
const float ColorSwatchCornerRadius = 4;
const float ColorSwatchStrokeSize = 4;
const float ColorSwatchWidth = 24;
#endif

DragImageRef fitDragImageToMaxSize(DragImageRef image, const IntSize& layoutSize, const IntSize& maxSize)
{
    float heightResizeRatio = 0.0f;
    float widthResizeRatio = 0.0f;
    float resizeRatio = -1.0f;
    IntSize originalSize = dragImageSize(image);

    if (layoutSize.width() > maxSize.width()) {
        widthResizeRatio = maxSize.width() / (float)layoutSize.width();
        resizeRatio = widthResizeRatio;
    }

    if (layoutSize.height() > maxSize.height()) {
        heightResizeRatio = maxSize.height() / (float)layoutSize.height();
        if ((resizeRatio < 0.0f) || (resizeRatio > heightResizeRatio))
            resizeRatio = heightResizeRatio;
    }

    if (layoutSize == originalSize)
        return resizeRatio > 0.0f ? scaleDragImage(image, FloatSize(resizeRatio, resizeRatio)) : image;

    // The image was scaled in the webpage so at minimum we must account for that scaling.
    float scaleX = layoutSize.width() / (float)originalSize.width();
    float scaleY = layoutSize.height() / (float)originalSize.height();
    if (resizeRatio > 0.0f) {
        scaleX *= resizeRatio;
        scaleY *= resizeRatio;
    }

    return scaleDragImage(image, FloatSize(scaleX, scaleY));
}

struct ScopedNodeDragEnabler {
    ScopedNodeDragEnabler(LocalFrame& frame, Node& node)
        : element(dynamicDowncast<Element>(node))
    {
        if (element)
            element->setBeingDragged(true);
        frame.protectedDocument()->updateLayout();
    }

    ~ScopedNodeDragEnabler()
    {
        if (element)
            element->setBeingDragged(false);
    }

    RefPtr<Element> element;
};

static DragImageRef createDragImageFromSnapshot(RefPtr<ImageBuffer> snapshot, Node* node)
{
    if (!snapshot)
        return nullptr;

    ImageOrientation orientation;
    if (node) {
        auto* elementRenderer = dynamicDowncast<RenderElement>(node->renderer());
        if (!elementRenderer)
            return nullptr;

        orientation = elementRenderer->imageOrientation();
    }

    auto image = BitmapImage::create(ImageBuffer::sinkIntoNativeImage(WTFMove(snapshot)));
    if (!image)
        return nullptr;
    return createDragImageFromImage(image.get(), orientation);
}

DragImageRef createDragImageForNode(LocalFrame& frame, Node& node)
{
    ScopedNodeDragEnabler enableDrag(frame, node);
    return createDragImageFromSnapshot(snapshotNode(frame, node, { { }, ImageBufferPixelFormat::BGRA8, DestinationColorSpace::SRGB() }), &node);
}

#if !PLATFORM(IOS_FAMILY) || !ENABLE(DRAG_SUPPORT)

DragImageRef createDragImageForSelection(LocalFrame& frame, TextIndicatorData&, bool forceBlackText)
{
    SnapshotOptions options { { }, ImageBufferPixelFormat::BGRA8, DestinationColorSpace::SRGB() };
    if (forceBlackText)
        options.flags.add(SnapshotFlags::ForceBlackText);
    return createDragImageFromSnapshot(snapshotSelection(frame, WTFMove(options)), nullptr);
}

#endif

struct ScopedFrameSelectionState {
    ScopedFrameSelectionState(LocalFrame& frame)
        : frame(frame)
    {
        if (auto* renderView = frame.contentRenderer())
            selection = renderView->selection().get();
    }

    ~ScopedFrameSelectionState()
    {
        if (auto* renderView = frame->contentRenderer()) {
            ASSERT(selection);
            renderView->selection().set(selection.value(), RenderSelection::RepaintMode::Nothing);
        }
    }

    const WeakRef<LocalFrame> frame;
    std::optional<RenderRange> selection;
};

#if !PLATFORM(IOS_FAMILY)

DragImageRef createDragImageForRange(LocalFrame& frame, const SimpleRange& range, bool forceBlackText)
{
    frame.protectedDocument()->updateLayout();
    RenderView* view = frame.contentRenderer();
    if (!view)
        return nullptr;

    // To snapshot the range, temporarily select it and take selection snapshot.
    Position start = makeDeprecatedLegacyPosition(range.start);
    Position candidate = start.downstream();
    if (candidate.deprecatedNode() && candidate.deprecatedNode()->renderer())
        start = candidate;

    Position end = makeDeprecatedLegacyPosition(range.end);
    candidate = end.upstream();
    if (candidate.deprecatedNode() && candidate.deprecatedNode()->renderer())
        end = candidate;

    if (start.isNull() || end.isNull() || start == end)
        return nullptr;

    const ScopedFrameSelectionState selectionState(frame);

    RenderObject* startRenderer = start.deprecatedNode()->renderer();
    RenderObject* endRenderer = end.deprecatedNode()->renderer();
    if (!startRenderer || !endRenderer)
        return nullptr;

    SnapshotOptions options { { SnapshotFlags::PaintSelectionOnly }, ImageBufferPixelFormat::BGRA8, DestinationColorSpace::SRGB() };
    if (forceBlackText)
        options.flags.add(SnapshotFlags::ForceBlackText);

    int startOffset = start.deprecatedEditingOffset();
    int endOffset = end.deprecatedEditingOffset();
    ASSERT(startOffset >= 0 && endOffset >= 0);
    view->selection().set({ startRenderer, endRenderer, static_cast<unsigned>(startOffset), static_cast<unsigned>(endOffset) }, RenderSelection::RepaintMode::Nothing);
    // We capture using snapshotFrameRect() because we fake up the selection using
    // FrameView but snapshotSelection() uses the selection from the Frame itself.
    return createDragImageFromSnapshot(snapshotFrameRect(frame, view->selection().boundsClippedToVisibleContent(), WTFMove(options)), nullptr);
}

#endif

DragImageRef createDragImageForImage(LocalFrame& frame, Node& node, IntRect& imageRect, IntRect& elementRect)
{
    ScopedNodeDragEnabler enableDrag(frame, node);

    RenderObject* renderer = node.renderer();
    if (!renderer)
        return nullptr;

    // Calculate image and element metrics for the client, then create drag image.
    LayoutRect topLevelRect;
    IntRect paintingRect = snappedIntRect(renderer->paintingRootRect(topLevelRect));

    if (paintingRect.isEmpty())
        return nullptr;

    elementRect = snappedIntRect(topLevelRect);
    imageRect = paintingRect;

    return createDragImageFromSnapshot(snapshotNode(frame, node, { { }, ImageBufferPixelFormat::BGRA8, DestinationColorSpace::SRGB() }), &node);
}

#if !PLATFORM(IOS_FAMILY) || !ENABLE(DRAG_SUPPORT)
DragImageRef platformAdjustDragImageForDeviceScaleFactor(DragImageRef image, float deviceScaleFactor)
{
    // Later code expects the drag image to be scaled by device's scale factor.
    return scaleDragImage(image, { deviceScaleFactor, deviceScaleFactor });
}
#endif

#if !PLATFORM(MAC)
const int linkDragBorderInset = 2;

IntPoint dragOffsetForLinkDragImage(DragImageRef dragImage)
{
    IntSize size = dragImageSize(dragImage);
    return { -size.width() / 2, -linkDragBorderInset };
}

FloatPoint anchorPointForLinkDragImage(DragImageRef dragImage)
{
    IntSize size = dragImageSize(dragImage);
    return { 0.5, static_cast<float>((size.height() - linkDragBorderInset) / size.height()) };
}
#endif

DragImage::DragImage()
    : m_dragImageRef { nullptr }
{
}

DragImage::DragImage(DragImageRef dragImageRef)
    : m_dragImageRef { dragImageRef }
{
}

DragImage::DragImage(DragImage&& other)
    : m_dragImageRef { std::exchange(other.m_dragImageRef, nullptr) }
    , m_indicatorData { WTFMove(other.m_indicatorData) }
    , m_visiblePath { WTFMove(other.m_visiblePath) }
{
}

DragImage& DragImage::operator=(DragImage&& other)
{
    if (m_dragImageRef)
        deleteDragImage(m_dragImageRef);

    m_dragImageRef = std::exchange(other.m_dragImageRef, nullptr);
    m_indicatorData = WTFMove(other.m_indicatorData);
    m_visiblePath = WTFMove(other.m_visiblePath);

    return *this;
}

DragImage::~DragImage()
{
    if (m_dragImageRef)
        deleteDragImage(m_dragImageRef);
}

#if !PLATFORM(COCOA) && !PLATFORM(GTK) && !PLATFORM(WIN)

IntSize dragImageSize(DragImageRef)
{
    notImplemented();
    return { 0, 0 };
}

void deleteDragImage(DragImageRef)
{
    notImplemented();
}

DragImageRef scaleDragImage(DragImageRef, FloatSize)
{
    notImplemented();
    return nullptr;
}

DragImageRef dissolveDragImageToFraction(DragImageRef, float)
{
    notImplemented();
    return nullptr;
}

DragImageRef createDragImageFromImage(Image*, ImageOrientation)
{
    notImplemented();
    return nullptr;
}

DragImageRef createDragImageIconForCachedImageFilename(const String&)
{
    notImplemented();
    return nullptr;
}

DragImageRef createDragImageForLink(Element&, URL&, const String&, TextIndicatorData&, float)
{
    notImplemented();
    return nullptr;
}

#endif

} // namespace WebCore

