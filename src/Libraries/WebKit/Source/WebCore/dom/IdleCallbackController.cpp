/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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
#include "IdleCallbackController.h"

#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "IdleDeadline.h"
#include "Page.h"
#include "Timer.h"
#include "WindowEventLoop.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IdleCallbackController);

IdleCallbackController::IdleCallbackController(Document& document)
    : m_document(document)
{

}

int IdleCallbackController::queueIdleCallback(Ref<IdleRequestCallback>&& callback, Seconds timeout)
{
    ++m_idleCallbackIdentifier;
    auto handle = m_idleCallbackIdentifier;

    bool hasTimeout = timeout > 0_s;
    m_idleRequestCallbacks.append({ handle, WTFMove(callback), hasTimeout ? std::optional { MonotonicTime::now() + timeout } : std::nullopt });

    if (hasTimeout) {
        Timer::schedule(timeout, [weakThis = WeakPtr { *this }, handle]() {
            if (!weakThis)
                return;
            RefPtr document = weakThis->m_document.get();
            if (!document)
                return;
            document->eventLoop().queueTask(TaskSource::IdleTask, [weakThis, handle]() {
                if (!weakThis)
                    return;
                weakThis->invokeIdleCallbackTimeout(handle);
            });
        });
    }

    if (RefPtr document = m_document.get())
        document->protectedWindowEventLoop()->scheduleIdlePeriod();

    return handle;
}

void IdleCallbackController::removeIdleCallback(int signedIdentifier)
{
    if (signedIdentifier <= 0)
        return;
    unsigned identifier = signedIdentifier;

    m_idleRequestCallbacks.removeAllMatching([identifier](auto& request) {
        return request.identifier == identifier;
    });

    m_runnableIdleCallbacks.removeAllMatching([identifier](auto& request) {
        return request.identifier == identifier;
    });
}

// https://w3c.github.io/requestidlecallback/#start-an-idle-period-algorithm
void IdleCallbackController::startIdlePeriod()
{
    for (auto& request : m_idleRequestCallbacks)
        m_runnableIdleCallbacks.append(WTFMove(request));
    m_idleRequestCallbacks.clear();

    if (m_runnableIdleCallbacks.isEmpty())
        return;

    while (invokeIdleCallbacks()) { }
}

void IdleCallbackController::queueTaskToInvokeIdleCallbacks()
{
    Ref document = *m_document;
    document->eventLoop().queueTask(TaskSource::IdleTask, [this, document] {
        RELEASE_ASSERT(document->idleCallbackController() == this);
        while (invokeIdleCallbacks()) { }
    });
}

// https://w3c.github.io/requestidlecallback/#invoke-idle-callbacks-algorithm
bool IdleCallbackController::invokeIdleCallbacks()
{
    RefPtr document = m_document.get();
    if (!document || !document->frame())
        return false;

    Ref windowEventLoop = document->windowEventLoop();
    // FIXME: Implement "if the user-agent believes it should end the idle period early due to newly scheduled high-priority work, return from the algorithm."

    auto now = MonotonicTime::now();
    auto deadline = windowEventLoop->computeIdleDeadline();
    if (now >= deadline || m_runnableIdleCallbacks.isEmpty())
        return false;

    auto request = m_runnableIdleCallbacks.takeFirst();
    auto idleDeadline = IdleDeadline::create(request.timeout && *request.timeout < now ? IdleDeadline::DidTimeout::Yes : IdleDeadline::DidTimeout::No);
    request.callback->handleEvent(idleDeadline.get());

    return !m_runnableIdleCallbacks.isEmpty();
}

// https://w3c.github.io/requestidlecallback/#dfn-invoke-idle-callback-timeout-algorithm
void IdleCallbackController::invokeIdleCallbackTimeout(unsigned identifier)
{
    if (!m_document)
        return;

    auto it = m_idleRequestCallbacks.findIf([identifier](auto& request) {
        return request.identifier == identifier;
    });

    if (it == m_idleRequestCallbacks.end())
        return;

    auto idleDeadline = IdleDeadline::create(IdleDeadline::DidTimeout::Yes);
    auto callback = WTFMove(it->callback);
    m_idleRequestCallbacks.remove(it);
    callback->handleEvent(idleDeadline.get());
}

} // namespace WebCore
