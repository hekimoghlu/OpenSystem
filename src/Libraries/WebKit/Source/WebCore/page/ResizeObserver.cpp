/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

#include "ResizeObserver.h"

#include "Element.h"
#include "InspectorInstrumentation.h"
#include "JSNodeCustom.h"
#include "Logging.h"
#include "ResizeObserverEntry.h"
#include "ResizeObserverOptions.h"
#include "WebCoreOpaqueRootInlines.h"
#include <JavaScriptCore/AbstractSlotVisitorInlines.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

Ref<ResizeObserver> ResizeObserver::create(Document& document, Ref<ResizeObserverCallback>&& callback)
{
    return adoptRef(*new ResizeObserver(document, { RefPtr<ResizeObserverCallback> { WTFMove(callback) } }));
}

Ref<ResizeObserver> ResizeObserver::createNativeObserver(Document& document, NativeResizeObserverCallback&& nativeCallback)
{
    return adoptRef(*new ResizeObserver(document, { WTFMove(nativeCallback) }));
}

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ResizeObserver);

ResizeObserver::ResizeObserver(Document& document, JSOrNativeResizeObserverCallback&& callback)
    : m_document(document)
    , m_JSOrNativeCallback(WTFMove(callback))
{
}

ResizeObserver::~ResizeObserver()
{
    disconnect();
    if (m_document)
        m_document->removeResizeObserver(*this);
}

void ResizeObserver::observeInternal(Element& target, const ResizeObserverBoxOptions boxOptions)
{
    ASSERT(!m_JSOrNativeCallback.valueless_by_exception());

    auto position = m_observations.findIf([&](auto& observation) {
        return observation->target() == &target;
    });

    if (position != notFound) {
        // The spec suggests unconditionally unobserving here, but that causes a test failure:
        // https://github.com/web-platform-tests/wpt/issues/30708
        if (m_observations[position]->observedBox() == boxOptions)
            return;

        unobserve(target);
    }

    auto& observerData = target.ensureResizeObserverData();
    observerData.observers.append(*this);

    m_observations.append(ResizeObservation::create(target, boxOptions));

    // Per the specification, we should dispatch at least one observation for the target. For this reason, we make sure to keep the
    // target alive until this first observation. This, in turn, will keep the ResizeObserver's JS wrapper alive via
    // isReachableFromOpaqueRoots(), so the callback stays alive.
    m_targetsWaitingForFirstObservation.append(target);

    if (m_document && isJSCallback()) {
        m_document->addResizeObserver(*this);
        m_document->scheduleRenderingUpdate(RenderingUpdateStep::ResizeObservations);
    }
}

// https://drafts.csswg.org/resize-observer/#dom-resizeobserver-observe
void ResizeObserver::observe(Element& target, const ResizeObserverOptions& options)
{
    observeInternal(target, options.box);
}

void ResizeObserver::observe(Element& target)
{
    observeInternal(target, ResizeObserverBoxOptions::ContentBox);
}

// https://drafts.csswg.org/resize-observer/#dom-resizeobserver-unobserve
void ResizeObserver::unobserve(Element& target)
{
    if (!removeTarget(target))
        return;

    removeObservation(target);
}

// https://drafts.csswg.org/resize-observer/#dom-resizeobserver-disconnect
void ResizeObserver::disconnect()
{
    removeAllTargets();
}

void ResizeObserver::targetDestroyed(Element& target)
{
    removeObservation(target);
}

size_t ResizeObserver::gatherObservations(size_t deeperThan)
{
    m_hasSkippedObservations = false;
    size_t minObservedDepth = maxElementDepth();
    for (const auto& observation : m_observations) {
        if (auto currentSizes = observation->elementSizeChanged()) {
            size_t depth = observation->targetElementDepth();
            if (depth > deeperThan) {
                observation->updateObservationSize(*currentSizes);

                LOG_WITH_STREAM(ResizeObserver, stream << "ResizeObserver " << this << " gatherObservations - recording observation " << observation.get());

                m_activeObservations.append(observation.get());
                m_activeObservationTargets.append(*observation->protectedTarget());
                minObservedDepth = std::min(depth, minObservedDepth);
            } else
                m_hasSkippedObservations = true;
        }
    }
    return minObservedDepth;
}

void ResizeObserver::deliverObservations()
{
    LOG_WITH_STREAM(ResizeObserver, stream << "ResizeObserver " << this << " deliverObservations");

    auto entries = m_activeObservations.map([](auto& observation) {
        ASSERT(observation->target());
        return ResizeObserverEntry::create(observation->target(), observation->computeContentRect(), observation->borderBoxSize(), observation->contentBoxSize());
    });
    m_activeObservations.clear();
    auto activeObservationTargets = std::exchange(m_activeObservationTargets, { });

    auto targetsWaitingForFirstObservation = std::exchange(m_targetsWaitingForFirstObservation, { });

    if (isNativeCallback()) {
        std::get<NativeResizeObserverCallback>(m_JSOrNativeCallback)(entries, *this);
        return;
    }

    // FIXME: The JSResizeObserver wrapper should be kept alive as long as the resize observer can fire events.
    ASSERT(isJSCallback());
    auto jsCallback = std::get<RefPtr<ResizeObserverCallback>>(m_JSOrNativeCallback);
    ASSERT(jsCallback->hasCallback());
    if (!jsCallback->hasCallback())
        return;

    RefPtr context = jsCallback->scriptExecutionContext();
    if (!context)
        return;

    InspectorInstrumentation::willFireObserverCallback(*context, "ResizeObserver"_s);
    jsCallback->handleEvent(*this, entries, *this);
    InspectorInstrumentation::didFireObserverCallback(*context);
}

bool ResizeObserver::isReachableFromOpaqueRoots(JSC::AbstractSlotVisitor& visitor) const
{
    for (auto& observation : m_observations) {
        if (auto* target = observation->target(); target && containsWebCoreOpaqueRoot(visitor, target))
            return true;
    }
    for (auto& target : m_activeObservationTargets) {
        SUPPRESS_UNCOUNTED_ARG {
            if (containsWebCoreOpaqueRoot(visitor, target.get()))
                return true;
        }
    }
    return !m_targetsWaitingForFirstObservation.isEmpty();
}

bool ResizeObserver::removeTarget(Element& target)
{
    auto* observerData = target.resizeObserverDataIfExists();
    if (!observerData)
        return false;

    auto& observers = observerData->observers;
    return observers.removeFirst(this);
}

void ResizeObserver::removeAllTargets()
{
    for (auto& observation : m_observations) {
        bool removed = removeTarget(*observation->protectedTarget());
        ASSERT_UNUSED(removed, removed);
    }
    m_activeObservationTargets.clear();
    m_activeObservations.clear();
    m_targetsWaitingForFirstObservation.clear();
    m_observations.clear();
}

bool ResizeObserver::removeObservation(const Element& target)
{
    m_targetsWaitingForFirstObservation.removeFirstMatching([&target](auto& pendingTarget) {
        return pendingTarget.ptr() == &target;
    });
    return m_observations.removeFirstMatching([&target](auto& observation) {
        return observation->target() == &target;
    });
}

bool ResizeObserver::isJSCallback()
{
    return std::holds_alternative<RefPtr<ResizeObserverCallback>>(m_JSOrNativeCallback);
}

bool ResizeObserver::isNativeCallback()
{
    return std::holds_alternative<NativeResizeObserverCallback>(m_JSOrNativeCallback);
}

ResizeObserverCallback* ResizeObserver::callbackConcurrently()
{
    return WTF::switchOn(m_JSOrNativeCallback,
    [] (const RefPtr<ResizeObserverCallback>& jsCallback) -> ResizeObserverCallback* {
        return jsCallback.get();
    },
    [] (const NativeResizeObserverCallback&) -> ResizeObserverCallback* {
        return nullptr;
    });
}

void ResizeObserver::resetObservationSize(Element& target)
{
    auto position = m_observations.findIf([&](auto& observation) {
        return observation->target() == &target;
    });

    if (position != notFound)
        m_observations[position]->resetObservationSize();
}

} // namespace WebCore
