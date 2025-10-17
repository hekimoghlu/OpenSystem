/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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

#include "FrameRateAligner.h"
#include "ReducedResolutionSeconds.h"
#include "ScrollAxis.h"
#include "TimelineScope.h"
#include "Timer.h"
#include <wtf/CancellableTask.h>
#include <wtf/CheckedRef.h>
#include <wtf/Markable.h>
#include <wtf/Seconds.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class AnimationTimeline;
class CSSTransition;
class Document;
class Element;
class ScrollTimeline;
class ViewTimeline;
class WeakPtrImplWithEventTargetData;
class WebAnimation;

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
class AcceleratedEffectStackUpdater;
#endif

struct ViewTimelineInsets;
struct TimelineMapAttachOperation {
    WeakPtr<Element, WeakPtrImplWithEventTargetData> element;
    AtomString name;
    Ref<WebAnimation> animation;
};

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(AnimationTimelinesController);
class AnimationTimelinesController final : public CanMakeCheckedPtr<AnimationTimelinesController> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(AnimationTimelinesController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AnimationTimelinesController);
public:
    explicit AnimationTimelinesController(Document&);
    ~AnimationTimelinesController();

    void addTimeline(AnimationTimeline&);
    void removeTimeline(AnimationTimeline&);
    void detachFromDocument();
    void updateAnimationsAndSendEvents(ReducedResolutionSeconds);

    std::optional<Seconds> currentTime();
    std::optional<FramesPerSecond> maximumAnimationFrameRate() const { return m_frameRateAligner.maximumFrameRate(); }
    std::optional<Seconds> timeUntilNextTickForAnimationsWithFrameRate(FramesPerSecond) const;

    WEBCORE_EXPORT void suspendAnimations();
    WEBCORE_EXPORT void resumeAnimations();
    bool animationsAreSuspended() const { return m_isSuspended; }

    void registerNamedScrollTimeline(const AtomString&, Element&, ScrollAxis);
    void registerNamedViewTimeline(const AtomString&, const Element&, ScrollAxis, ViewTimelineInsets&&);
    void unregisterNamedTimeline(const AtomString&, const Element&);
    void setTimelineForName(const AtomString&, const Element&, WebAnimation&);
    void updateNamedTimelineMapForTimelineScope(const TimelineScope&, const Element&);
    void updateTimelineForTimelineScope(const Ref<ScrollTimeline>&, const AtomString&);
    void unregisterNamedTimelinesAssociatedWithElement(const Element&);

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    AcceleratedEffectStackUpdater* existingAcceleratedEffectStackUpdater() const { return m_acceleratedEffectStackUpdater.get(); }
    AcceleratedEffectStackUpdater& acceleratedEffectStackUpdater();
#endif

private:

    ReducedResolutionSeconds liveCurrentTime() const;
    void cacheCurrentTime(ReducedResolutionSeconds);
    void maybeClearCachedCurrentTime();

    Vector<Ref<ScrollTimeline>>& timelinesForName(const AtomString&);
    Vector<WeakPtr<Element, WeakPtrImplWithEventTargetData>> relatedTimelineScopeElements(const AtomString&);
    void attachPendingOperations();
    bool isPendingTimelineAttachment(const WebAnimation&) const;
    void updateCSSAnimationsAssociatedWithNamedTimeline(const AtomString&);

    Ref<Document> protectedDocument() const { return m_document.get(); }

    Vector<TimelineMapAttachOperation> m_pendingAttachOperations;
    Vector<std::pair<TimelineScope, WeakPtr<Element, WeakPtrImplWithEventTargetData>>> m_timelineScopeEntries;
    UncheckedKeyHashMap<AtomString, Vector<Ref<ScrollTimeline>>> m_nameToTimelineMap;

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    std::unique_ptr<AcceleratedEffectStackUpdater> m_acceleratedEffectStackUpdater;
#endif

    UncheckedKeyHashMap<FramesPerSecond, ReducedResolutionSeconds> m_animationFrameRateToLastTickTimeMap;
    WeakHashSet<AnimationTimeline> m_timelines;
    TaskCancellationGroup m_currentTimeClearingTaskCancellationGroup;
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    FrameRateAligner m_frameRateAligner;
    Markable<Seconds, Seconds::MarkableTraits> m_cachedCurrentTime;
    bool m_isSuspended { false };
    bool m_waitingOnVMIdle { false };
};

} // namespace WebCore
