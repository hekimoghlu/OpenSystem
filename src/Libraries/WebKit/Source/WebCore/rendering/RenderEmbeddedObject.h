/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

#include "RenderWidget.h"

namespace WebCore {

class LayoutSize;
class MouseEvent;
class TextRun;

enum class PluginUnavailabilityReason : uint8_t {
    PluginMissing,
    PluginCrashed,
    PluginBlockedByContentSecurityPolicy,
    InsecurePluginVersion,
    UnsupportedPlugin,
    PluginTooSmall
};

// Renderer for embeds and objects, often, but not always, rendered via plug-ins.
// For example, <embed src="foo.html"> does not invoke a plug-in.
class RenderEmbeddedObject final : public RenderWidget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderEmbeddedObject);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderEmbeddedObject);
public:
    RenderEmbeddedObject(HTMLFrameOwnerElement&, RenderStyle&&);
    virtual ~RenderEmbeddedObject();

    PluginUnavailabilityReason pluginUnavailabilityReason() const { return m_pluginUnavailabilityReason; };
    WEBCORE_EXPORT void setPluginUnavailabilityReason(PluginUnavailabilityReason);

    bool isPluginUnavailable() const { return m_isPluginUnavailable; }

    void handleUnavailablePluginIndicatorEvent(Event*);

    bool requiresAcceleratedCompositing() const override;

    ScrollableArea* scrollableArea() const;
    bool usesAsyncScrolling() const;
    std::optional<ScrollingNodeID> scrollingNodeID() const;
    void willAttachScrollingNode();
    void didAttachScrollingNode();

    bool paintsContent() const final;

    void setHasShadowContent() { m_hasShadowContent = true; }

private:
    void paintReplaced(PaintInfo&, const LayoutPoint&) final;
    void paint(PaintInfo&, const LayoutPoint&) final;

    CursorDirective getCursor(const LayoutPoint&, Cursor&) const final;

    void layout() final;
    void willBeDestroyed() final;

    ASCIILiteral renderName() const final { return "RenderEmbeddedObject"_s; }

    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) final;

    bool scroll(ScrollDirection, ScrollGranularity, unsigned stepCount = 1, Element** stopElement = nullptr, RenderBox* startBox = nullptr, const IntPoint& wheelEventAbsolutePoint = IntPoint()) final;
    bool logicalScroll(ScrollLogicalDirection, ScrollGranularity, unsigned stepCount, Element** stopElement) final;

    void setUnavailablePluginIndicatorIsPressed(bool);
    bool isInUnavailablePluginIndicator(const MouseEvent&) const;
    bool isInUnavailablePluginIndicator(const FloatPoint&) const;
    void getReplacementTextGeometry(const LayoutPoint& accumulatedOffset, FloatRect& contentRect, FloatRect& indicatorRect, FloatRect& replacementTextRect, FloatRect& arrowRect, FontCascade&, TextRun&, float& textWidth) const;
    LayoutRect getReplacementTextGeometry(const LayoutPoint& accumulatedOffset) const;

    bool canHaveChildren() const override { return m_hasShadowContent; }

    bool m_isPluginUnavailable;
    PluginUnavailabilityReason m_pluginUnavailabilityReason;
    String m_unavailablePluginReplacementText;
    bool m_unavailablePluginIndicatorIsPressed;
    bool m_mouseDownWasInUnavailablePluginIndicator;
    String m_unavailabilityDescription;
    bool m_hasShadowContent { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderEmbeddedObject, isRenderEmbeddedObject())
