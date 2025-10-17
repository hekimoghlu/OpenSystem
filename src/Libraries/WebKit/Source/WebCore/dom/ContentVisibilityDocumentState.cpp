/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#include "ContentVisibilityDocumentState.h"

#include "ContentVisibilityAutoStateChangeEvent.h"
#include "DocumentInlines.h"
#include "DocumentTimeline.h"
#include "EventNames.h"
#include "FrameSelection.h"
#include "IntersectionObserverCallback.h"
#include "IntersectionObserverEntry.h"
#include "NodeRenderStyle.h"
#include "RenderElement.h"
#include "RenderStyleInlines.h"
#include "SimpleRange.h"
#include "StyleOriginatedAnimation.h"
#include "VisibleSelection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ContentVisibilityDocumentState);

class ContentVisibilityIntersectionObserverCallback final : public IntersectionObserverCallback {
public:
    static Ref<ContentVisibilityIntersectionObserverCallback> create(Document& document)
    {
        return adoptRef(*new ContentVisibilityIntersectionObserverCallback(document));
    }

private:
    ContentVisibilityIntersectionObserverCallback(Document& document)
        : IntersectionObserverCallback(&document)
    {
    }

    bool hasCallback() const final { return true; }

    CallbackResult<void> handleEvent(IntersectionObserver&, const Vector<Ref<IntersectionObserverEntry>>& entries, IntersectionObserver&) final
    {
        ASSERT(!entries.isEmpty());

        for (auto& entry : entries) {
            if (RefPtr element = entry->target())
                element->document().contentVisibilityDocumentState().updateViewportProximity(*element, entry->isIntersecting() ? ViewportProximity::Near : ViewportProximity::Far);
        }
        return { };
    }

    CallbackResult<void> handleEventRethrowingException(IntersectionObserver& thisObserver, const Vector<Ref<IntersectionObserverEntry>>& entries, IntersectionObserver& observer) final
    {
        return handleEvent(thisObserver, entries, observer);
    }
};

void ContentVisibilityDocumentState::observe(Element& element)
{
    Ref document = element.document();
    auto& state = document->contentVisibilityDocumentState();
    if (RefPtr intersectionObserver = state.intersectionObserver(document))
        intersectionObserver->observe(element);
}

void ContentVisibilityDocumentState::unobserve(Element& element)
{
    Ref document = element.document();
    auto& state = document->contentVisibilityDocumentState();
    if (RefPtr intersectionObserver = state.m_observer) {
        intersectionObserver->unobserve(element);
        state.removeViewportProximity(element);
    }
    element.setContentRelevancy({ });
}

IntersectionObserver* ContentVisibilityDocumentState::intersectionObserver(Document& document)
{
    if (!m_observer) {
        auto callback = ContentVisibilityIntersectionObserverCallback::create(document);
        IntersectionObserver::Init options { &document, { }, { } };
        auto observer = IntersectionObserver::create(document, WTFMove(callback), WTFMove(options));
        if (observer.hasException())
            return nullptr;
        m_observer = observer.releaseReturnValue();
    }
    return m_observer.get();
}

bool ContentVisibilityDocumentState::checkRelevancyOfContentVisibilityElement(Element& target, OptionSet<ContentRelevancy> relevancyToCheck) const
{
    auto oldRelevancy = target.contentRelevancy();
    OptionSet<ContentRelevancy> newRelevancy;
    if (oldRelevancy)
        newRelevancy = *oldRelevancy;
    auto setRelevancyValue = [&](ContentRelevancy reason, bool value) {
        if (value)
            newRelevancy.add(reason);
        else
            newRelevancy.remove(reason);
    };
    if (relevancyToCheck.contains(ContentRelevancy::OnScreen)) {
        auto viewportProximityIterator = m_elementViewportProximities.find(target);
        auto viewportProximity = ViewportProximity::Far;
        if (viewportProximityIterator != m_elementViewportProximities.end())
            viewportProximity = viewportProximityIterator->value;
        setRelevancyValue(ContentRelevancy::OnScreen, viewportProximity == ViewportProximity::Near);
    }

    if (relevancyToCheck.contains(ContentRelevancy::Focused))
        setRelevancyValue(ContentRelevancy::Focused, target.hasFocusWithin());

    auto targetContainsSelection = [](Element& target) {
        auto selectionRange = target.document().selection().selection().range();
        return selectionRange && intersects<ComposedTree>(*selectionRange, target);
    };

    if (relevancyToCheck.contains(ContentRelevancy::Selected))
        setRelevancyValue(ContentRelevancy::Selected, targetContainsSelection(target));

    auto hasTopLayerinSubtree = [](const Element& target) {
        for (Ref element : target.document().topLayerElements()) {
            if (element->isDescendantOf(target))
                return true;
        }
        return false;
    };
    if (relevancyToCheck.contains(ContentRelevancy::IsInTopLayer))
        setRelevancyValue(ContentRelevancy::IsInTopLayer, hasTopLayerinSubtree(target));

    if (oldRelevancy && oldRelevancy == newRelevancy)
        return false;

    auto wasSkippedContent = target.isRelevantToUser() ? IsSkippedContent::No : IsSkippedContent::Yes;
    target.setContentRelevancy(newRelevancy);
    auto isSkippedContent = target.isRelevantToUser() ? IsSkippedContent::No : IsSkippedContent::Yes;
    target.invalidateStyle();
    updateAnimations(target, wasSkippedContent, isSkippedContent);
    target.queueTaskKeepingThisNodeAlive(TaskSource::DOMManipulation, [&, isSkippedContent] {
        if (target.isConnected()) {
            ContentVisibilityAutoStateChangeEvent::Init init;
            init.skipped = isSkippedContent == IsSkippedContent::Yes;
            target.dispatchEvent(ContentVisibilityAutoStateChangeEvent::create(eventNames().contentvisibilityautostatechangeEvent, init));
        }
    });
    return true;
}

DidUpdateAnyContentRelevancy ContentVisibilityDocumentState::updateRelevancyOfContentVisibilityElements(OptionSet<ContentRelevancy> relevancyToCheck) const
{
    auto didUpdateAnyContentRelevancy = DidUpdateAnyContentRelevancy::No;
    for (auto& weakTarget : m_observer->observationTargets()) {
        if (RefPtr target = weakTarget.get()) {
            if (checkRelevancyOfContentVisibilityElement(*target, relevancyToCheck))
                didUpdateAnyContentRelevancy = DidUpdateAnyContentRelevancy::Yes;
        }
    }
    return didUpdateAnyContentRelevancy;
}

HadInitialVisibleContentVisibilityDetermination ContentVisibilityDocumentState::determineInitialVisibleContentVisibility() const
{
    if (!m_observer)
        return HadInitialVisibleContentVisibilityDetermination::No;
    Vector<Ref<Element>> elementsToCheck;
    for (auto& weakTarget : m_observer->observationTargets()) {
        if (RefPtr target = weakTarget.get()) {
            bool checkForInitialDetermination = !m_elementViewportProximities.contains(*target) && !target->isRelevantToUser();
            if (checkForInitialDetermination)
                elementsToCheck.append(target.releaseNonNull());
        }
    }
    auto hadInitialVisibleContentVisibilityDetermination = HadInitialVisibleContentVisibilityDetermination::No;
    if (!elementsToCheck.isEmpty()) {
        elementsToCheck.first()->protectedDocument()->updateIntersectionObservations({ m_observer });
        for (auto& element : elementsToCheck) {
            checkRelevancyOfContentVisibilityElement(element, { ContentRelevancy::OnScreen });
            if (element->isRelevantToUser())
                hadInitialVisibleContentVisibilityDetermination = HadInitialVisibleContentVisibilityDetermination::Yes;
        }
    }
    return hadInitialVisibleContentVisibilityDetermination;
}

// Make sure any skipped content we want to scroll to is in the viewport, so it can be actually
// scrolled to (i.e. the skipped content early exit in LocalFrameView::scrollRectToVisible does
// not apply anymore).
void ContentVisibilityDocumentState::updateContentRelevancyForScrollIfNeeded(const Element& scrollAnchor)
{
    if (!m_observer)
        return;
    auto findSkippedContentRoot = [](const Element& element) -> RefPtr<const Element> {
        RefPtr<const Element> found;
        if (element.renderer() && element.renderer()->isSkippedContent()) {
            for (RefPtr candidate = &element; candidate; candidate = candidate->parentElementInComposedTree()) {
                if (candidate->renderer() && candidate->renderStyle()->contentVisibility() == ContentVisibility::Auto)
                    found = candidate;
            }
        }
        return found;
    };

    if (RefPtr scrollAnchorRoot = findSkippedContentRoot(scrollAnchor)) {
        updateViewportProximity(*scrollAnchorRoot, ViewportProximity::Near);
        // Since we may not have determined initial visibility yet, force scheduling the content relevancy update.
        scrollAnchorRoot->protectedDocument()->scheduleContentRelevancyUpdate(ContentRelevancy::OnScreen);
        scrollAnchorRoot->protectedDocument()->updateRelevancyOfContentVisibilityElements();
    }
}

void ContentVisibilityDocumentState::updateViewportProximity(const Element& element, ViewportProximity viewportProximity)
{
    // No need to schedule content relevancy update for first time call, since
    // that will be handled by determineInitialVisibleContentVisibility.
    if (m_elementViewportProximities.contains(element))
        element.protectedDocument()->scheduleContentRelevancyUpdate(ContentRelevancy::OnScreen);
    m_elementViewportProximities.ensure(element, [] {
        return ViewportProximity::Far;
    }).iterator->value = viewportProximity;
}

void ContentVisibilityDocumentState::removeViewportProximity(const Element& element)
{
    m_elementViewportProximities.remove(element);
}

void ContentVisibilityDocumentState::updateAnimations(const Element& element, IsSkippedContent wasSkipped, IsSkippedContent becomesSkipped)
{
    if (wasSkipped == IsSkippedContent::No || becomesSkipped == IsSkippedContent::Yes)
        return;
    for (RefPtr animation : WebAnimation::instances()) {
        RefPtr styleOriginatedAnimation = dynamicDowncast<StyleOriginatedAnimation>(animation.releaseNonNull());
        if (!styleOriginatedAnimation)
            continue;
        auto owningElement = styleOriginatedAnimation->owningElement();
        if (!owningElement || !owningElement->element.isDescendantOrShadowDescendantOf(&element))
            continue;

        if (RefPtr timeline = styleOriginatedAnimation->timeline())
            timeline->animationTimingDidChange(*styleOriginatedAnimation);
    }
}

}
