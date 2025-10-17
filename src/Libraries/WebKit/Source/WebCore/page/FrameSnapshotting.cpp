/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#include "FrameSnapshotting.h"

#include "ColorBlending.h"
#include "Document.h"
#include "FloatRect.h"
#include "FrameSelection.h"
#include "GeometryUtilities.h"
#include "GraphicsContext.h"
#include "HostWindow.h"
#include "ImageBuffer.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Page.h"
#include "RenderAncestorIterator.h"
#include "RenderObject.h"
#include "RenderStyleInlines.h"
#include "Settings.h"

namespace WebCore {

struct ScopedFramePaintingState {
    ScopedFramePaintingState(LocalFrame& frame, Node* node)
        : frame(frame)
        , node(node)
        , paintBehavior(frame.view()->paintBehavior())
        , backgroundColor(frame.view()->baseBackgroundColor())
    {
        ASSERT(!node || node->renderer());
    }

    ~ScopedFramePaintingState()
    {
        frame.view()->setPaintBehavior(paintBehavior);
        frame.view()->setBaseBackgroundColor(backgroundColor);
        frame.view()->setNodeToDraw(nullptr);
    }

    const LocalFrame& frame;
    const Node* node;
    const OptionSet<PaintBehavior> paintBehavior;
    const Color backgroundColor;
};

RefPtr<ImageBuffer> snapshotFrameRect(LocalFrame& frame, const IntRect& imageRect, SnapshotOptions&& options)
{
    Vector<FloatRect> clipRects;
    return snapshotFrameRectWithClip(frame, imageRect, clipRects, WTFMove(options));
}

RefPtr<ImageBuffer> snapshotFrameRectWithClip(LocalFrame& frame, const IntRect& imageRect, const Vector<FloatRect>& clipRects, SnapshotOptions&& options)
{
    if (!frame.page())
        return nullptr;

    Ref document = *frame.document();
    document->updateLayout();

    LocalFrameView::SelectionInSnapshot shouldIncludeSelection = LocalFrameView::IncludeSelection;
    if (options.flags.contains(SnapshotFlags::ExcludeSelectionHighlighting))
        shouldIncludeSelection = LocalFrameView::ExcludeSelection;

    LocalFrameView::CoordinateSpaceForSnapshot coordinateSpace = LocalFrameView::DocumentCoordinates;
    if (options.flags.contains(SnapshotFlags::InViewCoordinates))
        coordinateSpace = LocalFrameView::ViewCoordinates;

    ScopedFramePaintingState state(frame, nullptr);

    auto paintBehavior = state.paintBehavior;
    if (options.flags.contains(SnapshotFlags::ForceBlackText))
        paintBehavior.add(PaintBehavior::ForceBlackText);
    if (options.flags.contains(SnapshotFlags::PaintSelectionOnly))
        paintBehavior.add(PaintBehavior::SelectionOnly);
    if (options.flags.contains(SnapshotFlags::PaintSelectionAndBackgroundsOnly))
        paintBehavior.add(PaintBehavior::SelectionAndBackgroundsOnly);
    if (options.flags.contains(SnapshotFlags::PaintEverythingExcludingSelection))
        paintBehavior.add(PaintBehavior::ExcludeSelection);
    if (options.flags.contains(SnapshotFlags::ExcludeReplacedContent))
        paintBehavior.add(PaintBehavior::ExcludeReplacedContent);

    // Other paint behaviors are set by paintContentsForSnapshot.
    frame.view()->setPaintBehavior(paintBehavior);

    float scaleFactor = frame.page()->deviceScaleFactor();
    if (options.flags.contains(SnapshotFlags::PaintWith3xBaseScale))
        scaleFactor = 3;

    if (frame.page()->delegatesScaling())
        scaleFactor *= frame.page()->pageScaleFactor();

    if (options.flags.contains(SnapshotFlags::PaintWithIntegralScaleFactor))
        scaleFactor = ceilf(scaleFactor);

    auto renderingMode = options.flags.contains(SnapshotFlags::Accelerated) ? RenderingMode::Accelerated : RenderingMode::Unaccelerated;
    auto purpose = options.flags.contains(SnapshotFlags::Shareable) ? RenderingPurpose::ShareableSnapshot : RenderingPurpose::Snapshot;
    auto hostWindow = (document->view() && document->view()->root()) ? document->view()->root()->hostWindow() : nullptr;

    auto buffer = ImageBuffer::create(imageRect.size(), renderingMode, purpose, scaleFactor, options.colorSpace, options.pixelFormat, hostWindow);
    if (!buffer)
        return nullptr;

    buffer->context().translate(-imageRect.location());

    if (!clipRects.isEmpty()) {
        Path clipPath;
        for (auto& rect : clipRects)
            clipPath.addRect(encloseRectToDevicePixels(rect, scaleFactor));
        buffer->context().clipPath(clipPath);
    }

    frame.view()->paintContentsForSnapshot(buffer->context(), imageRect, shouldIncludeSelection, coordinateSpace);
    return buffer;
}

RefPtr<ImageBuffer> snapshotSelection(LocalFrame& frame, SnapshotOptions&& options)
{
    auto& selection = frame.selection();

    if (!selection.isRange())
        return nullptr;

    FloatRect selectionBounds = selection.selectionBounds();

    // It is possible for the selection bounds to be empty; see https://bugs.webkit.org/show_bug.cgi?id=56645.
    if (selectionBounds.isEmpty())
        return nullptr;

    options.flags.add(SnapshotFlags::PaintSelectionOnly);
    return snapshotFrameRect(frame, enclosingIntRect(selectionBounds), WTFMove(options));
}

RefPtr<ImageBuffer> snapshotNode(LocalFrame& frame, Node& node, SnapshotOptions&& options)
{
    if (!node.renderer())
        return nullptr;

    ScopedFramePaintingState state(frame, &node);

    frame.view()->setBaseBackgroundColor(Color::transparentBlack);
    frame.view()->setNodeToDraw(&node);

    LayoutRect topLevelRect;
    return snapshotFrameRect(frame, snappedIntRect(node.renderer()->paintingRootRect(topLevelRect)), WTFMove(options));
}

static bool styleContainsComplexBackground(const RenderStyle& style)
{
    return style.hasBlendMode()
        || style.hasBackgroundImage()
#if HAVE(CORE_MATERIAL)
        || style.hasAppleVisualEffectRequiringBackdropFilter()
#endif
        || style.hasBackdropFilter();
}

Color estimatedBackgroundColorForRange(const SimpleRange& range, const LocalFrame& frame)
{
    auto estimatedBackgroundColor = frame.view() ? frame.view()->documentBackgroundColor() : Color::transparentBlack;

    RenderElement* renderer = nullptr;
    auto commonAncestor = commonInclusiveAncestor<ComposedTree>(range);
    while (commonAncestor) {
        if (auto* renderElement = dynamicDowncast<RenderElement>(commonAncestor->renderer())) {
            renderer = renderElement;
            break;
        }
        commonAncestor = commonAncestor->parentOrShadowHostElement();
    }
    
    auto boundingRectForRange = enclosingIntRect(unionRectIgnoringZeroRects(RenderObject::absoluteBorderAndTextRects(range, {
        RenderObject::BoundingRectBehavior::RespectClipping,
        RenderObject::BoundingRectBehavior::UseVisibleBounds,
        RenderObject::BoundingRectBehavior::IgnoreTinyRects,
    })));

    Vector<Color> parentRendererBackgroundColors;
    for (auto& ancestor : lineageOfType<RenderElement>(*renderer)) {
        auto absoluteBoundingBox = ancestor.absoluteBoundingBoxRect();
        auto& style = ancestor.style();
        if (!absoluteBoundingBox.contains(boundingRectForRange) || !style.hasBackground())
            continue;

        if (styleContainsComplexBackground(style))
            return estimatedBackgroundColor;

        auto visitedDependentBackgroundColor = style.visitedDependentColor(CSSPropertyBackgroundColor);
        if (visitedDependentBackgroundColor != Color::transparentBlack)
            parentRendererBackgroundColors.append(visitedDependentBackgroundColor);
    }
    parentRendererBackgroundColors.reverse();
    for (const auto& backgroundColor : parentRendererBackgroundColors)
        estimatedBackgroundColor = blendSourceOver(estimatedBackgroundColor, backgroundColor);

    return estimatedBackgroundColor;
}

} // namespace WebCore
