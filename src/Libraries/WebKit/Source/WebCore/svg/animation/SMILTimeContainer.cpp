/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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
#include "SMILTimeContainer.h"

#include "Document.h"
#include "ElementIterator.h"
#include "Page.h"
#include "SVGElementTypeHelpers.h"
#include "SVGSMILElement.h"
#include "SVGSVGElement.h"
#include "ScopedEventQueue.h"
#include "TypedElementDescendantIteratorInlines.h"

namespace WebCore {

static const Seconds SMILAnimationFrameDelay { 1_s / 60. };
static const Seconds SMILAnimationFrameThrottledDelay { 1_s / 30. };

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(SMILTimeContainer);

SMILTimeContainer::SMILTimeContainer(SVGSVGElement& owner)
    : m_timer(*this, &SMILTimeContainer::timerFired)
    , m_ownerSVGElement(owner)
{
}

void SMILTimeContainer::schedule(SVGSMILElement* animation, SVGElement* target, const QualifiedName& attributeName)
{
    ASSERT(animation->timeContainer() == this);
    ASSERT(target);
    ASSERT(animation->hasValidAttributeName());

    ElementAttributePair key(target, attributeName);
    auto& animations = m_scheduledAnimations.add(key, AnimationsVector()).iterator->value;
    ASSERT(!animations.contains(animation));
    animations.append(animation);

    SMILTime nextFireTime = animation->nextProgressTime();
    if (nextFireTime.isFinite())
        notifyIntervalsChanged();
}

void SMILTimeContainer::unschedule(SVGSMILElement* animation, SVGElement* target, const QualifiedName& attributeName)
{
    ASSERT(animation->timeContainer() == this);

    ElementAttributePair key(target, attributeName);
    auto& animations = m_scheduledAnimations.find(key)->value;
    ASSERT(animations.contains(animation));
    animations.removeFirst(animation);
}

void SMILTimeContainer::notifyIntervalsChanged()
{
    // Schedule updateAnimations() to be called asynchronously so multiple intervals
    // can change with updateAnimations() only called once at the end.
    startTimer(elapsed(), 0);
}

Seconds SMILTimeContainer::animationFrameDelay() const
{
    RefPtr page = m_ownerSVGElement->document().page();
    if (!page)
        return SMILAnimationFrameDelay;
    return (page->isLowPowerModeEnabled() || page->isAggressiveThermalMitigationEnabled()) ? SMILAnimationFrameThrottledDelay : SMILAnimationFrameDelay;
}

SMILTime SMILTimeContainer::elapsed() const
{
    if (!m_beginTime)
        return 0_s;
    if (isPaused())
        return m_accumulatedActiveTime;
    return MonotonicTime::now() + m_accumulatedActiveTime - m_resumeTime;
}

bool SMILTimeContainer::isActive() const
{
    return !!m_beginTime && !isPaused();
}

bool SMILTimeContainer::isPaused() const
{
    return !!m_pauseTime;
}

bool SMILTimeContainer::isStarted() const
{
    return !!m_beginTime;
}

void SMILTimeContainer::begin()
{
    if (isStarted())
        return;

    ASSERT(Page::nonUtilityPageCount());
    if (!Page::nonUtilityPageCount())
        return;

    MonotonicTime now = MonotonicTime::now();

    // If 'm_presetStartTime' is set, the timeline was modified via setElapsed() before the document began.
    // In this case pass on 'seekToTime=true' to updateAnimations().
    m_beginTime = m_resumeTime = now - m_presetStartTime;
    updateAnimations(SMILTime(m_presetStartTime), m_presetStartTime ? true : false);
    m_presetStartTime = 0_s;

    if (m_pauseTime) {
        m_pauseTime = now;
        m_timer.stop();
    }
}

void SMILTimeContainer::pause()
{
    ASSERT(!isPaused());

    m_pauseTime = MonotonicTime::now();
    if (m_beginTime) {
        m_accumulatedActiveTime += m_pauseTime - m_resumeTime;
        m_timer.stop();
    }
}

void SMILTimeContainer::resume()
{
    ASSERT(isPaused());
    ASSERT(Page::nonUtilityPageCount());
    if (!Page::nonUtilityPageCount())
        return;

    m_resumeTime = MonotonicTime::now();
    m_pauseTime = MonotonicTime();
    startTimer(elapsed(), 0);
}

void SMILTimeContainer::setElapsed(SMILTime time)
{
    ASSERT(Page::nonUtilityPageCount());
    if (!Page::nonUtilityPageCount())
        return;

    // If the document didn't begin yet, record a new start time, we'll seek to once its possible.
    if (!m_beginTime) {
        m_presetStartTime = Seconds(time.value());
        return;
    }

    if (m_beginTime)
        m_timer.stop();

    MonotonicTime now = MonotonicTime::now();
    m_beginTime = now - Seconds { time.value() };

    if (m_pauseTime) {
        m_resumeTime = m_pauseTime = now;
        m_accumulatedActiveTime = Seconds(time.value());
    } else
        m_resumeTime = m_beginTime;

    processScheduledAnimations([](auto& animation) {
        animation.reset();
    });

    updateAnimations(time, true);
}

void SMILTimeContainer::startTimer(SMILTime elapsed, SMILTime fireTime, SMILTime minimumDelay)
{
    if (!m_beginTime || isPaused())
        return;

    if (!fireTime.isFinite())
        return;

    ASSERT(Page::nonUtilityPageCount());
    if (!Page::nonUtilityPageCount())
        return;

    SMILTime delay = std::max(fireTime - elapsed, minimumDelay);
    m_timer.startOneShot(1_s * delay.value());
}

void SMILTimeContainer::timerFired()
{
    ASSERT(!!m_beginTime);
    ASSERT(!m_pauseTime);
    updateAnimations(elapsed());
}

void SMILTimeContainer::updateDocumentOrderIndexes()
{
    unsigned timingElementCount = 0;

    for (Ref smilElement : descendantsOfType<SVGSMILElement>(Ref { m_ownerSVGElement.get() }))
        smilElement->setDocumentOrderIndex(timingElementCount++);

    m_documentOrderIndexesDirty = false;
}

struct PriorityCompare {
    PriorityCompare(SMILTime elapsed) : m_elapsed(elapsed) {}
    bool operator()(SVGSMILElement* a, SVGSMILElement* b)
    {
        // FIXME: This should also consider possible timing relations between the elements.
        SMILTime aBegin = a->intervalBegin();
        SMILTime bBegin = b->intervalBegin();
        // Frozen elements need to be prioritized based on their previous interval.
        aBegin = a->isFrozen() && m_elapsed < aBegin ? a->previousIntervalBegin() : aBegin;
        bBegin = b->isFrozen() && m_elapsed < bBegin ? b->previousIntervalBegin() : bBegin;
        if (aBegin == bBegin)
            return a->documentOrderIndex() < b->documentOrderIndex();
        return aBegin < bBegin;
    }
    SMILTime m_elapsed;
};

void SMILTimeContainer::sortByPriority(AnimationsVector& animations, SMILTime elapsed)
{
    if (m_documentOrderIndexesDirty)
        updateDocumentOrderIndexes();
    std::sort(animations.begin(), animations.end(), PriorityCompare(elapsed));
}

void SMILTimeContainer::processScheduledAnimations(const Function<void(SVGSMILElement&)>& callback)
{
    for (auto& animations : copyToVector(m_scheduledAnimations.values())) {
        for (RefPtr animation : animations)
            callback(*animation);
    }
}

void SMILTimeContainer::updateAnimations(SMILTime elapsed, bool seekToTime)
{
    ASSERT(Page::nonUtilityPageCount());
    if (!Page::nonUtilityPageCount())
        return;

    // Don't mutate the DOM while updating the animations.
    EventQueueScope scope;

    processScheduledAnimations([](auto& animation) {
        if (!animation.hasConditionsConnected())
            animation.connectConditions();
    });

    AnimationsVector animationsToApply;
    SMILTime earliestFireTime = SMILTime::unresolved();

    for (auto& animations : copyToVector(m_scheduledAnimations.values())) {
        // Sort according to priority. Elements with later begin time have higher priority.
        // In case of a tie, document order decides.
        // FIXME: This should also consider timing relationships between the elements. Dependents
        // have higher priority.
        sortByPriority(animations, elapsed);

        RefPtr<SVGSMILElement> firstAnimation;
        for (RefPtr animation : animations) {
            ASSERT(animation->timeContainer() == this);
            ASSERT(animation->targetElement());
            ASSERT(animation->hasValidAttributeName());

            // Results are accumulated to the first animation that animates and contributes to a particular element/attribute pair.
            if (!firstAnimation) {
                if (!animation->hasValidAttributeType())
                    return;
                firstAnimation = animation;
            }

            // This will calculate the contribution from the animation and add it to the resultsElement.
            if (!animation->progress(elapsed, *firstAnimation, seekToTime) && firstAnimation == animation)
                firstAnimation = nullptr;

            SMILTime nextFireTime = animation->nextProgressTime();
            if (nextFireTime.isFinite())
                earliestFireTime = std::min(nextFireTime, earliestFireTime);
        }

        if (firstAnimation)
            animationsToApply.append(firstAnimation.get());
    }

    // Apply results to target elements.
    for (RefPtr animation : animationsToApply)
        animation->applyResultsToTarget();

    startTimer(elapsed, earliestFireTime, animationFrameDelay());
}

}
