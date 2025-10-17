/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#include "AnimationTimeline.h"
#include "Element.h"
#include "ScrollAxis.h"
#include "ScrollTimelineOptions.h"
#include "Styleable.h"
#include <wtf/Ref.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AnimationTimelinesController;
class Document;
class Element;
class RenderStyle;
class ScrollableArea;

struct TimelineRange;

TextStream& operator<<(TextStream&, Scroller);

class ScrollTimeline : public AnimationTimeline {
public:
    static Ref<ScrollTimeline> create(Document&, ScrollTimelineOptions&& = { });
    static Ref<ScrollTimeline> create(const AtomString&, ScrollAxis);
    static Ref<ScrollTimeline> create(Scroller, ScrollAxis);

    virtual Element* source() const;
    void setSource(Element*);
    void setSource(const Styleable&);

    ScrollAxis axis() const { return m_axis; }
    void setAxis(ScrollAxis axis) { m_axis = axis; }

    const AtomString& name() const { return m_name; }
    void setName(const AtomString& name) { m_name = name; }

    AnimationTimeline::ShouldUpdateAnimationsAndSendEvents documentWillUpdateAnimationsAndSendEvents() override;

    AnimationTimelinesController* controller() const override;

    std::optional<WebAnimationTime> currentTime() override;
    TimelineRange defaultRange() const override;
    WeakPtr<Element, WeakPtrImplWithEventTargetData> timelineScopeDeclaredElement() const { return m_timelineScopeElement; }
    void setTimelineScopeElement(const Element&);
    void clearTimelineScopeDeclaredElement() { m_timelineScopeElement = nullptr; }

    virtual std::pair<WebAnimationTime, WebAnimationTime> intervalForAttachmentRange(const TimelineRange&) const;

protected:
    explicit ScrollTimeline(const AtomString&, ScrollAxis);

    struct Data {
        float scrollOffset { 0 };
        float rangeStart { 0 };
        float rangeEnd { 0 };
    };
    static float floatValueForOffset(const Length&, float);
    virtual Data computeTimelineData() const;

    static ScrollableArea* scrollableAreaForSourceRenderer(const RenderElement*, Document&);

    struct ResolvedScrollDirection {
        bool isVertical;
        bool isReversed;
    };
    std::optional<ResolvedScrollDirection> resolvedScrollDirection() const;

private:
    explicit ScrollTimeline();
    explicit ScrollTimeline(Scroller, ScrollAxis);

    bool isScrollTimeline() const final { return true; }

    void animationTimingDidChange(WebAnimation&) override;

    void removeTimelineFromDocument(Element*);

    struct CurrentTimeData {
        float scrollOffset { 0 };
        float maxScrollOffset { 0 };
    };

    void cacheCurrentTime();

    WeakStyleable m_source;
    ScrollAxis m_axis { ScrollAxis::Block };
    AtomString m_name;
    Scroller m_scroller { Scroller::Self };
    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_timelineScopeElement;
    CurrentTimeData m_cachedCurrentTimeData { };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ANIMATION_TIMELINE(ScrollTimeline, isScrollTimeline())
