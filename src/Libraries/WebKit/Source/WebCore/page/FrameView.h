/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

#include "ScrollView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class Frame;
enum class RenderAsTextFlag : uint16_t;

class FrameView : public ScrollView {
    WTF_MAKE_TZONE_ALLOCATED(FrameView);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(FrameView);
public:
    enum class Type : bool { Local, Remote };
    virtual Type viewType() const = 0;
    virtual void writeRenderTreeAsText(TextStream&, OptionSet<RenderAsTextFlag>) = 0;
    virtual Frame& frame() const = 0;

    WEBCORE_EXPORT int headerHeight() const final;
    WEBCORE_EXPORT int footerHeight() const final;

    WEBCORE_EXPORT float topContentInset(TopContentInsetType = TopContentInsetType::WebCoreContentInset) const final;

    float visibleContentScaleFactor() const final;

    HostWindow* hostWindow() const final;
    WEBCORE_EXPORT void invalidateRect(const IntRect&) final;
    bool isActive() const final;
    bool forceUpdateScrollbarsOnMainThreadForPerformanceTesting() const final;
    IntRect scrollableAreaBoundingBox(bool* = nullptr) const final;

    void scrollbarStyleChanged(ScrollbarStyle, bool forceUpdate) override;

    // Coordinate systems:
    //
    // "View"
    //     Top left is top left of the FrameView/ScrollView/Widget. Size is Widget::boundsRect().size().
    //
    // "TotalContents"
    //    Relative to ScrollView's scrolled contents, including headers and footers. Size is totalContentsSize().
    //
    // "Contents"
    //    Relative to ScrollView's scrolled contents, excluding headers and footers, so top left is top left of the scroll view's
    //    document, and size is contentsSize().
    //
    // "Absolute"
    //    Relative to the document's scroll origin (non-zero for RTL documents), but affected by page zoom and page scale. Mostly used
    //    in rendering code.
    //
    // "Document"
    //    Relative to the document's scroll origin, but not affected by page zoom or page scale. Size is equivalent to CSS pixel dimensions.
    //    FIXME: some uses are affected by page zoom (e.g. layout and visual viewports).
    //
    // "Client"
    //    Relative to the visible part of the document (or, more strictly, the layout viewport rect), and with the same scaling
    //    as Document coordinates, i.e. matching CSS pixels. Affected by scroll origin.
    //
    // "LayoutViewport"
    //    Similar to client coordinates, but affected by page zoom (but not page scale).
    //

    // Methods to convert points and rects between the coordinate space of the renderer, and this view.
    WEBCORE_EXPORT IntRect convertFromRendererToContainingView(const RenderElement*, const IntRect&) const;
    WEBCORE_EXPORT IntRect convertFromContainingViewToRenderer(const RenderElement*, const IntRect&) const;
    WEBCORE_EXPORT FloatRect convertFromContainingViewToRenderer(const RenderElement*, const FloatRect&) const;
    WEBCORE_EXPORT IntPoint convertFromRendererToContainingView(const RenderElement*, const IntPoint&) const;
    WEBCORE_EXPORT FloatPoint convertFromRendererToContainingView(const RenderElement*, const FloatPoint&) const;
    WEBCORE_EXPORT IntPoint convertFromContainingViewToRenderer(const RenderElement*, const IntPoint&) const;

    // Override ScrollView methods to do point conversion via renderers, in order to take transforms into account.
    IntRect convertToContainingView(const IntRect&) const final;
    IntRect convertFromContainingView(const IntRect&) const final;
    FloatRect convertFromContainingView(const FloatRect&) const final;
    IntPoint convertToContainingView(const IntPoint&) const final;
    FloatPoint convertToContainingView(const FloatPoint&) const final;
    IntPoint convertFromContainingView(const IntPoint&) const final;

private:
    ScrollableArea* enclosingScrollableArea() const final;

    bool scrollAnimatorEnabled() const final;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::FrameView)
static bool isType(const WebCore::Widget& widget) { return widget.isLocalFrameView() || widget.isRemoteFrameView(); }
SPECIALIZE_TYPE_TRAITS_END()
