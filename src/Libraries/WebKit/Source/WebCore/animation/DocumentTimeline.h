/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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

#include "AnimationFrameRate.h"
#include "AnimationTimeline.h"
#include "DocumentTimelineOptions.h"
#include "ExceptionOr.h"
#include "Timer.h"
#include <wtf/Ref.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class AnimationTimelinesController;
class AnimationEventBase;
class CustomEffectCallback;
class Document;
class AnimationTimelinesController;
class Element;
class RenderBoxModelObject;
class RenderElement;
class WeakPtrImplWithEventTargetData;
class WebAnimation;

struct CustomAnimationOptions;

class DocumentTimeline final : public AnimationTimeline
{
public:
    static Ref<DocumentTimeline> create(Document&);
    static Ref<DocumentTimeline> create(Document&, DocumentTimelineOptions&&);

    virtual ~DocumentTimeline();

    Document* document() const { return m_document.get(); }

    std::optional<WebAnimationTime> currentTime() override;
    ExceptionOr<Ref<WebAnimation>> animate(Ref<CustomEffectCallback>&&, std::optional<std::variant<double, CustomAnimationOptions>>&&);

    void animationTimingDidChange(WebAnimation&) override;
    void removeAnimation(WebAnimation&) override;
    void transitionDidComplete(Ref<CSSTransition>&&);

    void animationAcceleratedRunningStateDidChange(WebAnimation&);
    void detachFromDocument() override;

    void enqueueAnimationEvent(AnimationEventBase&);
    bool hasPendingAnimationEventForAnimation(const WebAnimation&) const;
    
    ShouldUpdateAnimationsAndSendEvents documentWillUpdateAnimationsAndSendEvents() override;
    void removeReplacedAnimations();
    AnimationEvents prepareForPendingAnimationEventsDispatch();
    void documentDidUpdateAnimationsAndSendEvents();
    void styleOriginatedAnimationsWereCreated();

    WEBCORE_EXPORT Seconds animationInterval() const;
    void suspendAnimations() override;
    void resumeAnimations() override;
    WEBCORE_EXPORT unsigned numberOfActiveAnimationsForTesting() const;
    WEBCORE_EXPORT Vector<std::pair<String, double>> acceleratedAnimationsForElement(Element&) const;    
    WEBCORE_EXPORT unsigned numberOfAnimationTimelineInvalidationsForTesting() const;

    Seconds convertTimelineTimeToOriginRelativeTime(Seconds) const;

    std::optional<FramesPerSecond> maximumFrameRate() const;

private:
    DocumentTimeline(Document&, Seconds);

    bool isDocumentTimeline() const final { return true; }

    AnimationTimelinesController* controller() const override;
    void applyPendingAcceleratedAnimations();
    void scheduleInvalidationTaskIfNeeded();
    void scheduleAnimationResolution();
    void clearTickScheduleTimer();
    void internalUpdateAnimationsAndSendEvents();
    void scheduleNextTick();
    bool animationCanBeRemoved(WebAnimation&);
    bool shouldRunUpdateAnimationsAndSendEventsIgnoringSuspensionState() const;

    Timer m_tickScheduleTimer;
    UncheckedKeyHashSet<RefPtr<WebAnimation>> m_acceleratedAnimationsPendingRunningStateChange;
    AnimationEvents m_pendingAnimationEvents;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    Seconds m_originTime;
    unsigned m_numberOfAnimationTimelineInvalidationsForTesting { 0 };
    bool m_animationResolutionScheduled { false };
    bool m_shouldScheduleAnimationResolutionForNewPendingEvents { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_ANIMATION_TIMELINE(DocumentTimeline, isDocumentTimeline())
