/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "ActiveDOMObject.h"

#include "Document.h"
#include "Event.h"
#include "EventLoop.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static inline ScriptExecutionContext* suitableScriptExecutionContext(ScriptExecutionContext* scriptExecutionContext)
{
    // For detached documents, make sure we observe their context document instead.
    if (auto* document = dynamicDowncast<Document>(scriptExecutionContext))
        return &document->contextDocument();
    return scriptExecutionContext;
}

inline ActiveDOMObject::ActiveDOMObject(ScriptExecutionContext* context, CheckedScriptExecutionContextType)
    : ContextDestructionObserver(context)
{
    ASSERT(!is<Document>(context) || &downcast<Document>(context)->contextDocument() == downcast<Document>(context));
    if (!context)
        return;

    ASSERT(context->isContextThread());
    context->didCreateActiveDOMObject(*this);
}

ActiveDOMObject::ActiveDOMObject(ScriptExecutionContext* scriptExecutionContext)
    : ActiveDOMObject(suitableScriptExecutionContext(scriptExecutionContext), CheckedScriptExecutionContext)
{
}

ActiveDOMObject::ActiveDOMObject(Document* document)
    : ActiveDOMObject(document ? &document->contextDocument() : nullptr, CheckedScriptExecutionContext)
{
}

ActiveDOMObject::ActiveDOMObject(Document& document)
    : ActiveDOMObject(&document.contextDocument(), CheckedScriptExecutionContext)
{
}

ActiveDOMObject::~ActiveDOMObject()
{
    ASSERT(canCurrentThreadAccessThreadLocalData(m_creationThread));

    // ActiveDOMObject may be inherited by a sub-class whose life-cycle
    // exceeds that of the associated ScriptExecutionContext. In those cases,
    // m_scriptExecutionContext would/should have been nullified by
    // ContextDestructionObserver::contextDestroyed() (which we implement /
    // inherit). Hence, we should ensure that this is not 0 before use it
    // here.

    RefPtr<ScriptExecutionContext> context = scriptExecutionContext();
    if (!context)
        return;

    ASSERT(m_suspendIfNeededWasCalled);
    ASSERT(context->isContextThread());
    context->willDestroyActiveDOMObject(*this);
}

void ActiveDOMObject::suspendIfNeeded()
{
#if ASSERT_ENABLED
    ASSERT(!m_suspendIfNeededWasCalled);
    m_suspendIfNeededWasCalled = true;
#endif
    if (RefPtr<ScriptExecutionContext> context = scriptExecutionContext())
        context->suspendActiveDOMObjectIfNeeded(*this);
}

#if ASSERT_ENABLED

void ActiveDOMObject::assertSuspendIfNeededWasCalled() const
{
    ASSERT(m_suspendIfNeededWasCalled);
}

#endif // ASSERT_ENABLED

void ActiveDOMObject::didMoveToNewDocument(Document& newDocument)
{
    if (RefPtr context = scriptExecutionContext())
        context->willDestroyActiveDOMObject(*this);
    Ref newScriptExecutionContext = newDocument.contextDocument();
    observeContext(newScriptExecutionContext.ptr());
    newScriptExecutionContext->didCreateActiveDOMObject(*this);
}

void ActiveDOMObject::suspend(ReasonForSuspension)
{
}

void ActiveDOMObject::resume()
{
}

void ActiveDOMObject::stop()
{
}

bool ActiveDOMObject::isContextStopped() const
{
    return !scriptExecutionContext() || scriptExecutionContext()->activeDOMObjectsAreStopped();
}

bool ActiveDOMObject::isAllowedToRunScript() const
{
    return scriptExecutionContext() && !scriptExecutionContext()->activeDOMObjectsAreStopped() && !scriptExecutionContext()->activeDOMObjectsAreSuspended();
}

void ActiveDOMObject::queueTaskInEventLoop(TaskSource source, Function<void ()>&& function)
{
    RefPtr<ScriptExecutionContext> context = scriptExecutionContext();
    if (!context)
        return;
    context->eventLoop().queueTask(source, WTFMove(function));
}

class ActiveDOMObjectEventDispatchTask : public EventLoopTask {
    WTF_MAKE_TZONE_ALLOCATED(ActiveDOMObjectEventDispatchTask);
public:
    ActiveDOMObjectEventDispatchTask(TaskSource source, EventLoopTaskGroup& group, ActiveDOMObject& object, Function<void()>&& dispatchEvent)
        : EventLoopTask(source, group)
        , m_object(object)
        , m_dispatchEvent(WTFMove(dispatchEvent))
    {
        ++m_object->m_pendingActivityInstanceCount;
    }

    ~ActiveDOMObjectEventDispatchTask()
    {
        ASSERT(m_object->m_pendingActivityInstanceCount);
        --m_object->m_pendingActivityInstanceCount;
    }

    void execute() final
    {
        // If this task executes after the script execution context has been stopped, don't
        // actually dispatch the event.
        if (m_object->isAllowedToRunScript())
            m_dispatchEvent();
    }

private:
    Ref<ActiveDOMObject> m_object;
    Function<void()> m_dispatchEvent;
};

WTF_MAKE_TZONE_ALLOCATED_IMPL(ActiveDOMObjectEventDispatchTask);

void ActiveDOMObject::queueTaskToDispatchEventInternal(EventTarget& target, TaskSource source, Ref<Event>&& event)
{
    ASSERT(!event->target() || &target == event->target());
    RefPtr context = scriptExecutionContext();
    if (!context)
        return;
    auto& eventLoopTaskGroup = context->eventLoop();
    auto task = makeUnique<ActiveDOMObjectEventDispatchTask>(source, eventLoopTaskGroup, *this, [target = Ref { target }, event = WTFMove(event)] {
        target->dispatchEvent(event);
    });
    eventLoopTaskGroup.queueTask(WTFMove(task));
}

void ActiveDOMObject::queueCancellableTaskToDispatchEventInternal(EventTarget& target, TaskSource source, TaskCancellationGroup& cancellationGroup, Ref<Event>&& event)
{
    ASSERT(!event->target() || &target == event->target());
    RefPtr context = scriptExecutionContext();
    if (!context)
        return;
    auto& eventLoopTaskGroup = context->eventLoop();
    auto task = makeUnique<ActiveDOMObjectEventDispatchTask>(source, eventLoopTaskGroup, *this, CancellableTask(cancellationGroup, [target = Ref { target }, event = WTFMove(event)] {
        target->dispatchEvent(event);
    }));
    eventLoopTaskGroup.queueTask(WTFMove(task));
}

} // namespace WebCore
