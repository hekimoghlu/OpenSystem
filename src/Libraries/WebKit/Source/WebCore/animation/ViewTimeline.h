/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

#include "CSSNumericValue.h"
#include "CSSPrimitiveValue.h"
#include "ScrollTimeline.h"
#include "ViewTimelineOptions.h"
#include <wtf/Ref.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

namespace Style {
class BuilderState;
}

class Element;

struct TimelineRange;

class ViewTimeline final : public ScrollTimeline {
public:
    static ExceptionOr<Ref<ViewTimeline>> create(Document&, ViewTimelineOptions&& = { });
    static Ref<ViewTimeline> create(const AtomString&, ScrollAxis, ViewTimelineInsets&&);

    const Element* subject() const { return m_subject.get(); }
    void setSubject(const Element*);

    const ViewTimelineInsets& insets() const { return m_insets; }
    void setInsets(ViewTimelineInsets&& insets) { m_insets = WTFMove(insets); }

    Ref<CSSNumericValue> startOffset() const;
    Ref<CSSNumericValue> endOffset() const;

    AnimationTimeline::ShouldUpdateAnimationsAndSendEvents documentWillUpdateAnimationsAndSendEvents() override;
    AnimationTimelinesController* controller() const override;

    const RenderBox* sourceScrollerRenderer() const;
    Element* source() const override;
    TimelineRange defaultRange() const final;

private:
    ScrollTimeline::Data computeTimelineData() const final;
    std::pair<WebAnimationTime, WebAnimationTime> intervalForAttachmentRange(const TimelineRange&) const final;

    explicit ViewTimeline(ScrollAxis);
    explicit ViewTimeline(const AtomString&, ScrollAxis, ViewTimelineInsets&&);

    bool isViewTimeline() const final { return true; }

    struct CurrentTimeData {
        float scrollOffset { 0 };
        float scrollContainerSize { 0 };
        float subjectOffset { 0 };
        float subjectSize { 0 };
        float insetStart { 0 };
        float insetEnd { 0 };
    };

    void cacheCurrentTime();

    struct SpecifiedViewTimelineInsets {
        RefPtr<CSSPrimitiveValue> start;
        RefPtr<CSSPrimitiveValue> end;
    };

    ExceptionOr<SpecifiedViewTimelineInsets> validateSpecifiedInsets(const ViewTimelineInsetValue, const Document&);

    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_subject;
    std::optional<SpecifiedViewTimelineInsets> m_specifiedInsets;
    ViewTimelineInsets m_insets;
    CurrentTimeData m_cachedCurrentTimeData { };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ANIMATION_TIMELINE(ViewTimeline, isViewTimeline())
