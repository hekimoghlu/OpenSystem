/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#include "WorkerAnimationController.h"

#if ENABLE(OFFSCREEN_CANVAS_IN_WORKERS)

#include "Performance.h"
#include "RequestAnimationFrameCallback.h"
#include "WorkerGlobalScope.h"
#include <wtf/Ref.h>

namespace WebCore {

Ref<WorkerAnimationController> WorkerAnimationController::create(WorkerGlobalScope& workerGlobalScope)
{
    auto controller = adoptRef(*new WorkerAnimationController(workerGlobalScope));
    controller->suspendIfNeeded();
    return controller;
}

WorkerAnimationController::WorkerAnimationController(WorkerGlobalScope& workerGlobalScope)
    : ActiveDOMObject(&workerGlobalScope)
    , m_workerGlobalScope(workerGlobalScope)
    , m_animationTimer(*this, &WorkerAnimationController::animationTimerFired)
{
}

WorkerAnimationController::~WorkerAnimationController()
{
    ASSERT(!hasPendingActivity());
}

bool WorkerAnimationController::virtualHasPendingActivity() const
{
    return m_animationTimer.isActive();
}

void WorkerAnimationController::stop()
{
    m_animationTimer.stop();
    m_animationCallbacks.clear();
}

void WorkerAnimationController::suspend(ReasonForSuspension)
{
    m_savedIsActive = hasPendingActivity();
    stop();
}

void WorkerAnimationController::resume()
{
    if (m_savedIsActive) {
        m_savedIsActive = false;
        scheduleAnimation();
    }
}

WorkerAnimationController::CallbackId WorkerAnimationController::requestAnimationFrame(Ref<RequestAnimationFrameCallback>&& callback)
{
    // FIXME: There's a lot of missing throttling behaviour that's present on DOMDocument
    WorkerAnimationController::CallbackId callbackId = ++m_nextAnimationCallbackId;
    callback->m_firedOrCancelled = false;
    callback->m_id = callbackId;
    m_animationCallbacks.append(WTFMove(callback));

    scheduleAnimation();

    return callbackId;
}

void WorkerAnimationController::cancelAnimationFrame(CallbackId callbackId)
{
    for (size_t i = 0; i < m_animationCallbacks.size(); ++i) {
        auto& callback = m_animationCallbacks[i];
        if (callback->m_id == callbackId) {
            callback->m_firedOrCancelled = true;
            m_animationCallbacks.remove(i);
            return;
        }
    }
}

void WorkerAnimationController::scheduleAnimation()
{
    if (m_animationTimer.isActive())
        return;

    Seconds animationInterval = RequestAnimationFrameCallback::fullSpeedAnimationInterval;
    Seconds scheduleDelay = std::max(animationInterval - Seconds::fromMilliseconds(m_workerGlobalScope.performance().now() - m_lastAnimationFrameTimestamp), 0_s);

    m_animationTimer.startOneShot(scheduleDelay);
}

void WorkerAnimationController::animationTimerFired()
{
    m_lastAnimationFrameTimestamp = m_workerGlobalScope.performance().now();
    serviceRequestAnimationFrameCallbacks(m_lastAnimationFrameTimestamp);
}

void WorkerAnimationController::serviceRequestAnimationFrameCallbacks(DOMHighResTimeStamp timestamp)
{
    if (!m_animationCallbacks.size())
        return;

    // First, generate a list of callbacks to consider. Callbacks registered from this point
    // on are considered only for the "next" frame, not this one.
    CallbackList callbacks(m_animationCallbacks);

    for (auto& callback : callbacks) {
        if (callback->m_firedOrCancelled)
            continue;
        callback->m_firedOrCancelled = true;
        callback->handleEvent(timestamp);
    }

    // Remove any callbacks we fired from the list of pending callbacks.
    m_animationCallbacks.removeAllMatching([](auto& callback) {
        return callback->m_firedOrCancelled;
    });

    if (m_animationCallbacks.size())
        scheduleAnimation();
}

} // namespace WebCore

#endif
