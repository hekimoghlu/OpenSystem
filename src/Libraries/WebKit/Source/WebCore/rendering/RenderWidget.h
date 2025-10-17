/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

#include "HTMLFrameOwnerElement.h"
#include "OverlapTestRequestClient.h"
#include "RenderReplaced.h"
#include "Widget.h"

namespace WebCore {

class RemoteFrame;

class WidgetHierarchyUpdatesSuspensionScope {
public:
    WidgetHierarchyUpdatesSuspensionScope()
    {
        s_widgetHierarchyUpdateSuspendCount++;
    }
    ~WidgetHierarchyUpdatesSuspensionScope()
    {
        ASSERT(s_widgetHierarchyUpdateSuspendCount);
        if (s_widgetHierarchyUpdateSuspendCount == 1 && s_haveScheduledWidgetToMove)
            moveWidgets();
        s_widgetHierarchyUpdateSuspendCount--;
    }

    static bool isSuspended() { return s_widgetHierarchyUpdateSuspendCount; }
    static void scheduleWidgetToMove(Widget&, LocalFrameView*);

private:
    using WidgetToParentMap = UncheckedKeyHashMap<RefPtr<Widget>, SingleThreadWeakPtr<LocalFrameView>>;
    static WidgetToParentMap& widgetNewParentMap();

    WEBCORE_EXPORT void moveWidgets();
    WEBCORE_EXPORT static unsigned s_widgetHierarchyUpdateSuspendCount;
    WEBCORE_EXPORT static bool s_haveScheduledWidgetToMove;
};

inline void WidgetHierarchyUpdatesSuspensionScope::scheduleWidgetToMove(Widget& widget, LocalFrameView* frame)
{
    s_haveScheduledWidgetToMove = true;
    widgetNewParentMap().set(&widget, frame);
}

class RenderWidget : public RenderReplaced, private OverlapTestRequestClient, public RefCounted<RenderWidget> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderWidget);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderWidget);
public:
    virtual ~RenderWidget();

    HTMLFrameOwnerElement& frameOwnerElement() const { return downcast<HTMLFrameOwnerElement>(nodeForNonAnonymous()); }

    Widget* widget() const { return m_widget.get(); }
    RefPtr<Widget> protectedWidget() const { return m_widget; }
    WEBCORE_EXPORT void setWidget(RefPtr<Widget>&&);

    static RenderWidget* find(const Widget&);

    enum class ChildWidgetState { Valid, Destroyed };
    ChildWidgetState updateWidgetPosition() WARN_UNUSED_RETURN;
    WEBCORE_EXPORT IntRect windowClipRect() const;

    virtual bool requiresAcceleratedCompositing() const;

    RemoteFrame* remoteFrame() const;

protected:
    RenderWidget(Type, HTMLFrameOwnerElement&, RenderStyle&&);

    void willBeDestroyed() override;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;
    void layout() override;
    void paint(PaintInfo&, const LayoutPoint&) override;
    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;
    bool requiresLayer() const override;

private:
    void element() const = delete;

    bool needsPreferredWidthsRecalculation() const final;
    RenderBox* embeddedContentBox() const final;

    void setSelectionState(HighlightState) final;
    void setOverlapTestResult(bool) final;

    bool setWidgetGeometry(const LayoutRect&);
    bool updateWidgetGeometry();

    void paintContents(PaintInfo&, const LayoutPoint&);

    RefPtr<Widget> m_widget;
    IntRect m_clipRect; // The rectangle needs to remain correct after scrolling, so it is stored in content view coordinates, and not clipped to window.
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderWidget, isRenderWidget())
