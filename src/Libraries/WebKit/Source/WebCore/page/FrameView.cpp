/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#include "FrameView.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "FocusController.h"
#include "Frame.h"
#include "HTMLFrameOwnerElement.h"
#include "Page.h"
#include "RenderElement.h"
#include "RenderLayer.h"
#include "RenderLayerScrollableArea.h"
#include "RenderWidget.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(FrameView);

int FrameView::headerHeight() const
{
    Ref frame = this->frame();
    if (!frame->isMainFrame())
        return 0;
    Page* page = frame->page();
    return page ? page->headerHeight() : 0;
}

int FrameView::footerHeight() const
{
    Ref frame = this->frame();
    if (!frame->isMainFrame())
        return 0;
    Page* page = frame->page();
    return page ? page->footerHeight() : 0;
}

float FrameView::topContentInset(TopContentInsetType contentInsetTypeToReturn) const
{
    if (platformWidget() && contentInsetTypeToReturn == TopContentInsetType::WebCoreOrPlatformContentInset)
        return platformTopContentInset();

    Ref frame = this->frame();
    if (!frame->isMainFrame())
        return 0;

    Page* page = frame->page();
    return page ? page->topContentInset() : 0;
}

float FrameView::visibleContentScaleFactor() const
{
    Ref frame = this->frame();
    if (!frame->isMainFrame())
        return 1;

    Page* page = frame->page();
    // FIXME: This !delegatesScaling() is confusing, and the opposite behavior to Frame::frameScaleFactor().
    // This function should probably be renamed to delegatedPageScaleFactor().
    if (!page || !page->delegatesScaling())
        return 1;

    return page->pageScaleFactor();
}

bool FrameView::isActive() const
{
    Page* page = frame().page();
    return page && page->focusController().isActive();
}

ScrollableArea* FrameView::enclosingScrollableArea() const
{
    Ref frame = this->frame();
    if (frame->isMainFrame())
        return nullptr;

    auto* ownerElement = frame->ownerElement();
    if (!ownerElement)
        return nullptr;

    auto* ownerRenderer = ownerElement->renderer();
    if (!ownerRenderer)
        return nullptr;

    auto* layer = ownerRenderer->enclosingLayer();
    if (!layer)
        return nullptr;

    auto* enclosingScrollableLayer = layer->enclosingScrollableLayer(IncludeSelfOrNot::IncludeSelf, CrossFrameBoundaries::No);
    if (!enclosingScrollableLayer)
        return nullptr;

    return enclosingScrollableLayer->scrollableArea();
}

void FrameView::invalidateRect(const IntRect& rect)
{
    Ref frame = this->frame();
    if (!parent()) {
        if (auto* page = frame->page())
            page->chrome().invalidateContentsAndRootView(rect);
        return;
    }

    CheckedPtr renderer = frame->ownerRenderer();
    if (!renderer)
        return;

    IntRect repaintRect = rect;
    repaintRect.moveBy(roundedIntPoint(renderer->contentBoxLocation()));
    renderer->repaintRectangle(repaintRect);
}

bool FrameView::forceUpdateScrollbarsOnMainThreadForPerformanceTesting() const
{
    Page* page = frame().page();
    return page && page->settings().scrollingPerformanceTestingEnabled();
}

IntRect FrameView::scrollableAreaBoundingBox(bool*) const
{
    RefPtr ownerRenderer = frame().ownerRenderer();
    if (!ownerRenderer)
        return frameRect();

    return ownerRenderer->absoluteContentQuad().enclosingBoundingBox();
}

HostWindow* FrameView::hostWindow() const
{
    auto* page = frame().page();
    return page ? &page->chrome() : nullptr;
}

void FrameView::scrollbarStyleChanged(ScrollbarStyle newStyle, bool forceUpdate)
{
    Ref frame = this->frame();
    if (!frame->isMainFrame())
        return;

    if (Page* page = frame->page())
        page->chrome().client().recommendedScrollbarStyleDidChange(newStyle);

    ScrollView::scrollbarStyleChanged(newStyle, forceUpdate);
}

bool FrameView::scrollAnimatorEnabled() const
{
    if (auto* page = frame().page())
        return page->settings().scrollAnimatorEnabled();

    return false;
}

IntRect FrameView::convertFromRendererToContainingView(const RenderElement* renderer, const IntRect& rendererRect) const
{
    IntRect rect = snappedIntRect(enclosingLayoutRect(renderer->localToAbsoluteQuad(FloatRect(rendererRect)).boundingBox()));

    return contentsToView(rect);
}

IntRect FrameView::convertFromContainingViewToRenderer(const RenderElement* renderer, const IntRect& viewRect) const
{
    IntRect rect = viewToContents(viewRect);

    // FIXME: we don't have a way to map an absolute rect down to a local quad, so just
    // move the rect for now.
    rect.setLocation(roundedIntPoint(renderer->absoluteToLocal(rect.location(), UseTransforms)));
    return rect;
}

FloatRect FrameView::convertFromContainingViewToRenderer(const RenderElement* renderer, const FloatRect& viewRect) const
{
    FloatRect rect = viewToContents(viewRect);

    return (renderer->absoluteToLocalQuad(rect)).boundingBox();
}

IntPoint FrameView::convertFromRendererToContainingView(const RenderElement* renderer, const IntPoint& rendererPoint) const
{
    IntPoint point = roundedIntPoint(renderer->localToAbsolute(rendererPoint, UseTransforms));

    return contentsToView(point);
}

FloatPoint FrameView::convertFromRendererToContainingView(const RenderElement* renderer, const FloatPoint& rendererPoint) const
{
    return contentsToView(renderer->localToAbsolute(rendererPoint, UseTransforms));
}

IntPoint FrameView::convertFromContainingViewToRenderer(const RenderElement* renderer, const IntPoint& viewPoint) const
{
    IntPoint point = viewPoint;

    // Convert from FrameView coords into page ("absolute") coordinates.
    if (!delegatesScrollingToNativeView())
        point = viewToContents(point);

    return roundedIntPoint(renderer->absoluteToLocal(point, UseTransforms));
}

IntRect FrameView::convertToContainingView(const IntRect& localRect) const
{
    if (auto* parentScrollView = parent()) {
        if (auto* parentView = dynamicDowncast<FrameView>(*parentScrollView)) {
            // Get our renderer in the parent view
            RenderWidget* renderer = frame().ownerRenderer();
            if (!renderer)
                return localRect;

            auto rect = localRect;
            rect.moveBy(roundedIntPoint(renderer->contentBoxLocation()));
            return parentView->convertFromRendererToContainingView(renderer, rect);
        }
        return Widget::convertToContainingView(localRect);
    }
    return localRect;
}

IntRect FrameView::convertFromContainingView(const IntRect& parentRect) const
{
    if (auto* parentScrollView = parent()) {
        if (auto* parentView = dynamicDowncast<FrameView>(*parentScrollView)) {
            // Get our renderer in the parent view
            RenderWidget* renderer = frame().ownerRenderer();
            if (!renderer)
                return parentRect;

            auto rect = parentView->convertFromContainingViewToRenderer(renderer, parentRect);
            rect.moveBy(-roundedIntPoint(renderer->contentBoxLocation()));
            return rect;
        }
        return Widget::convertFromContainingView(parentRect);
    }
    return parentRect;
}

FloatRect FrameView::convertFromContainingView(const FloatRect& parentRect) const
{
    if (auto* parentScrollView = parent()) {
        if (auto* parentView = dynamicDowncast<FrameView>(*parentScrollView)) {
            // Get our renderer in the parent view
            RenderWidget* renderer = frame().ownerRenderer();
            if (!renderer)
                return parentRect;

            auto rect = parentView->convertFromContainingViewToRenderer(renderer, parentRect);
            rect.moveBy(-renderer->contentBoxLocation());
            return rect;
        }
        return Widget::convertFromContainingView(parentRect);
    }
    return parentRect;
}

IntPoint FrameView::convertToContainingView(const IntPoint& localPoint) const
{
    if (auto* parentScrollView = parent()) {
        if (auto* parentView = dynamicDowncast<FrameView>(*parentScrollView)) {
            // Get our renderer in the parent view
            RenderWidget* renderer = frame().ownerRenderer();
            if (!renderer)
                return localPoint;

            auto point = localPoint;
            point.moveBy(roundedIntPoint(renderer->contentBoxLocation()));
            return parentView->convertFromRendererToContainingView(renderer, point);
        }
        return Widget::convertToContainingView(localPoint);
    }
    return localPoint;
}

FloatPoint FrameView::convertToContainingView(const FloatPoint& localPoint) const
{
    if (auto* parentScrollView = parent()) {
        if (auto* parentView = dynamicDowncast<FrameView>(*parentScrollView)) {
            // Get our renderer in the parent view
            RenderWidget* renderer = frame().ownerRenderer();
            if (!renderer)
                return localPoint;

            auto point = localPoint;
            point.moveBy(renderer->contentBoxLocation());
            return parentView->convertFromRendererToContainingView(renderer, point);
        }
        return Widget::convertToContainingView(localPoint);
    }
    return localPoint;
}

IntPoint FrameView::convertFromContainingView(const IntPoint& parentPoint) const
{
    if (auto* parentScrollView = parent()) {
        if (auto* parentView = dynamicDowncast<FrameView>(*parentScrollView)) {
            // Get our renderer in the parent view
            RenderWidget* renderer = frame().ownerRenderer();
            if (!renderer)
                return parentPoint;

            auto point = parentView->convertFromContainingViewToRenderer(renderer, parentPoint);
            point.moveBy(-roundedIntPoint(renderer->contentBoxLocation()));
            return point;
        }
        return Widget::convertFromContainingView(parentPoint);
    }
    return parentPoint;
}

}
