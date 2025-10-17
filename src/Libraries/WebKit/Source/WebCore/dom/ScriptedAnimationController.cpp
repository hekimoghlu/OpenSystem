/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#include "ScriptedAnimationController.h"

#include "InspectorInstrumentation.h"
#include "Logging.h"
#include "OpportunisticTaskScheduler.h"
#include "Page.h"
#include "RequestAnimationFrameCallback.h"
#include "Settings.h"
#include "UserGestureIndicator.h"
#include <wtf/SystemTracing.h>

namespace WebCore {

ScriptedAnimationController::ScriptedAnimationController(Document& document)
    : m_document(document)
{
}

ScriptedAnimationController::~ScriptedAnimationController() = default;

void ScriptedAnimationController::suspend()
{
    ++m_suspendCount;
}

void ScriptedAnimationController::resume()
{
    // It would be nice to put an ASSERT(m_suspendCount > 0) here, but in WK1 resume() can be called
    // even when suspend hasn't (if a tab was created in the background).
    if (m_suspendCount > 0)
        --m_suspendCount;

    if (!m_suspendCount && m_callbackDataList.size())
        scheduleAnimation();
}

Page* ScriptedAnimationController::page() const
{
    return m_document ? m_document->page() : nullptr;
}

Seconds ScriptedAnimationController::interval() const
{
    return preferredScriptedAnimationInterval();
}

Seconds ScriptedAnimationController::preferredScriptedAnimationInterval() const
{
    auto* page = this->page();
    if (!page)
        return FullSpeedAnimationInterval;

    return preferredFrameInterval(throttlingReasons(), page->displayNominalFramesPerSecond(), page->settings().preferPageRenderingUpdatesNear60FPSEnabled());
}

OptionSet<ThrottlingReason> ScriptedAnimationController::throttlingReasons() const
{
    if (auto* page = this->page())
        return page->throttlingReasons() | m_throttlingReasons;

    return m_throttlingReasons;
}

bool ScriptedAnimationController::isThrottledRelativeToPage() const
{
    if (auto* page = this->page())
        return preferredScriptedAnimationInterval() > page->preferredRenderingUpdateInterval();
    return false;
}

bool ScriptedAnimationController::shouldRescheduleRequestAnimationFrame(ReducedResolutionSeconds timestamp) const
{
    LOG_WITH_STREAM(RequestAnimationFrame, stream << "ScriptedAnimationController::shouldRescheduleRequestAnimationFrame - throttled relative to page " << isThrottledRelativeToPage()
        << ", last delta " << (timestamp - m_lastAnimationFrameTimestamp).milliseconds() << "ms, preferred interval " << preferredScriptedAnimationInterval().milliseconds() << ")");

    return timestamp <= m_lastAnimationFrameTimestamp || (isThrottledRelativeToPage() && (timestamp - m_lastAnimationFrameTimestamp < preferredScriptedAnimationInterval()));
}

ScriptedAnimationController::CallbackId ScriptedAnimationController::registerCallback(Ref<RequestAnimationFrameCallback>&& callback)
{
    CallbackId callbackId = ++m_nextCallbackId;
    callback->m_firedOrCancelled = false;
    callback->m_id = callbackId;
    RefPtr<ImminentlyScheduledWorkScope> workScope;
    if (RefPtr page = this->page())
        workScope = page->opportunisticTaskScheduler().makeScheduledWorkScope();
    m_callbackDataList.append({ WTFMove(callback), UserGestureIndicator::currentUserGesture(), WTFMove(workScope) });

    if (RefPtr document = m_document.get())
        InspectorInstrumentation::didRequestAnimationFrame(*document, callbackId);

    if (!m_suspendCount)
        scheduleAnimation();
    return callbackId;
}

void ScriptedAnimationController::cancelCallback(CallbackId callbackId)
{
    bool cancelled = m_callbackDataList.removeFirstMatching([callbackId](auto& data) {
        if (data.callback->m_id != callbackId)
            return false;
        data.callback->m_firedOrCancelled = true;
        return true;
    });

    if (cancelled && m_document)
        InspectorInstrumentation::didCancelAnimationFrame(*protectedDocument(), callbackId);
}

void ScriptedAnimationController::serviceRequestAnimationFrameCallbacks(ReducedResolutionSeconds timestamp)
{
    if (!m_callbackDataList.size() || m_suspendCount)
        return;

    if (shouldRescheduleRequestAnimationFrame(timestamp)) {
        LOG_WITH_STREAM(RequestAnimationFrame, stream << "ScriptedAnimationController::serviceRequestAnimationFrameCallbacks - rescheduling (page update interval " << (page() ? page()->preferredRenderingUpdateInterval() : 0_s) << ", raf interval " << preferredScriptedAnimationInterval() << ")");
        scheduleAnimation();
        return;
    }
    
    TraceScope tracingScope(RAFCallbackStart, RAFCallbackEnd);

    auto highResNowMs = std::round(1000 * timestamp.seconds());

    LOG_WITH_STREAM(RequestAnimationFrame, stream << "ScriptedAnimationController::serviceRequestAnimationFrameCallbacks at " << highResNowMs << " (throttling reasons " << throttlingReasons() << ", preferred interval " << preferredScriptedAnimationInterval().milliseconds() << "ms)");

    // First, generate a list of callbacks to consider.  Callbacks registered from this point
    // on are considered only for the "next" frame, not this one.
    Vector<CallbackData> callbackDataList(m_callbackDataList);

    // Invoking callbacks may detach elements from our document, which clears the document's
    // reference to us, so take a defensive reference.
    Ref protectedThis { *this };
    Ref document = *m_document;

    for (auto& [callback, userGestureTokenToForward, scheduledWorkScope] : callbackDataList) {
        if (callback->m_firedOrCancelled)
            continue;
        callback->m_firedOrCancelled = true;

        if (userGestureTokenToForward && userGestureTokenToForward->hasExpired(UserGestureToken::maximumIntervalForUserGestureForwarding))
            userGestureTokenToForward = nullptr;
        UserGestureIndicator gestureIndicator(userGestureTokenToForward);

        auto identifier = callback->m_id;
        InspectorInstrumentation::willFireAnimationFrame(document, identifier);
        callback->handleEvent(highResNowMs);
        InspectorInstrumentation::didFireAnimationFrame(document, identifier);
    }

    // Remove any callbacks we fired from the list of pending callbacks.
    m_callbackDataList.removeAllMatching([](auto& data) {
        return data.callback->m_firedOrCancelled;
    });

    m_lastAnimationFrameTimestamp = timestamp;

    if (m_callbackDataList.size())
        scheduleAnimation();
}

void ScriptedAnimationController::scheduleAnimation()
{
    if (RefPtr page = this->page())
        page->scheduleRenderingUpdate(RenderingUpdateStep::AnimationFrameCallbacks);
}

RefPtr<Document> ScriptedAnimationController::protectedDocument()
{
    return m_document.get();
}

}
