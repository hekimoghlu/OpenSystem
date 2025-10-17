/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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
#include "RejectedPromiseTracker.h"

#include "EventNames.h"
#include "EventTarget.h"
#include "JSDOMGlobalObject.h"
#include "JSDOMPromise.h"
#include "Node.h"
#include "PromiseRejectionEvent.h"
#include "ScriptExecutionContext.h"
#include <JavaScriptCore/Exception.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSGlobalObject.h>
#include <JavaScriptCore/JSPromise.h>
#include <JavaScriptCore/ScriptCallStack.h>
#include <JavaScriptCore/ScriptCallStackFactory.h>
#include <JavaScriptCore/Strong.h>
#include <JavaScriptCore/StrongInlines.h>
#include <JavaScriptCore/Weak.h>
#include <JavaScriptCore/WeakGCMapInlines.h>
#include <JavaScriptCore/WeakInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;
using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RejectedPromiseTracker);

class UnhandledPromise {
    WTF_MAKE_NONCOPYABLE(UnhandledPromise);
public:
    UnhandledPromise(JSDOMGlobalObject& globalObject, JSPromise& promise, RefPtr<ScriptCallStack>&& stack)
        : m_promise(DOMPromise::create(globalObject, promise))
        , m_stack(WTFMove(stack))
    {
    }

    UnhandledPromise(UnhandledPromise&&) = default;

    ScriptCallStack* callStack()
    {
        return m_stack.get();
    }

    DOMPromise& promise()
    {
        return m_promise.get();
    }

private:
    Ref<DOMPromise> m_promise;
    RefPtr<ScriptCallStack> m_stack;
};


RejectedPromiseTracker::RejectedPromiseTracker(ScriptExecutionContext& context, JSC::VM& vm)
    : m_context(context)
    , m_outstandingRejectedPromises(vm)
{
}

RejectedPromiseTracker::~RejectedPromiseTracker() = default;

static RefPtr<ScriptCallStack> createScriptCallStackFromReason(JSGlobalObject& lexicalGlobalObject, JSValue reason)
{
    // Always capture a stack from the exception if this rejection was an exception.
    if (auto* exception = lexicalGlobalObject.vm().lastException()) {
        if (exception->value() == reason)
            return createScriptCallStackFromException(&lexicalGlobalObject, exception);
    }

    // Otherwise, only capture a stack if a debugger is open.
    if (lexicalGlobalObject.debugger())
        return createScriptCallStack(&lexicalGlobalObject);

    return nullptr;
}

void RejectedPromiseTracker::promiseRejected(JSDOMGlobalObject& globalObject, JSPromise& promise)
{
    // https://html.spec.whatwg.org/multipage/webappapis.html#the-hostpromiserejectiontracker-implementation

    JSValue reason = promise.result(globalObject.vm());
    m_aboutToBeNotifiedRejectedPromises.append(UnhandledPromise { globalObject, promise, createScriptCallStackFromReason(globalObject, reason) });
}

void RejectedPromiseTracker::promiseHandled(JSDOMGlobalObject& globalObject, JSPromise& promise)
{
    // https://html.spec.whatwg.org/multipage/webappapis.html#the-hostpromiserejectiontracker-implementation

    bool removed = m_aboutToBeNotifiedRejectedPromises.removeFirstMatching([&] (UnhandledPromise& unhandledPromise) {
        auto& domPromise = unhandledPromise.promise();
        if (domPromise.isSuspended())
            return false;
        return domPromise.promise() == &promise;
    });
    if (removed)
        return;

    if (!m_outstandingRejectedPromises.remove(&promise))
        return;

    m_context->postTask([this, rejectedPromise = DOMPromise::create(globalObject, promise)] (ScriptExecutionContext&) mutable {
        reportRejectionHandled(WTFMove(rejectedPromise));
    });
}

void RejectedPromiseTracker::processQueueSoon()
{
    // https://html.spec.whatwg.org/multipage/webappapis.html#notify-about-rejected-promises

    if (m_aboutToBeNotifiedRejectedPromises.isEmpty())
        return;

    Vector<UnhandledPromise> items = WTFMove(m_aboutToBeNotifiedRejectedPromises);
    m_context->postTask([this, items = WTFMove(items)] (ScriptExecutionContext&) mutable {
        reportUnhandledRejections(WTFMove(items));
    });
}

void RejectedPromiseTracker::reportUnhandledRejections(Vector<UnhandledPromise>&& unhandledPromises)
{
    // https://html.spec.whatwg.org/multipage/webappapis.html#unhandled-promise-rejections

    Ref vm = m_context->vm();
    JSC::JSLockHolder lock(vm);

    for (auto& unhandledPromise : unhandledPromises) {
        auto& domPromise = unhandledPromise.promise();
        if (domPromise.isSuspended())
            continue;
        auto& lexicalGlobalObject = *domPromise.globalObject();
        auto& promise = *domPromise.promise();

        if (promise.isHandled(vm))
            continue;

        PromiseRejectionEvent::Init initializer;
        initializer.cancelable = true;
        initializer.promise = &domPromise;
        initializer.reason = promise.result(vm);

        Ref event = PromiseRejectionEvent::create(eventNames().unhandledrejectionEvent, initializer);
        RefPtr target = m_context->errorEventTarget();
        target->dispatchEvent(event);

        if (!event->defaultPrevented())
            m_context->reportUnhandledPromiseRejection(lexicalGlobalObject, promise, unhandledPromise.callStack());

        if (!promise.isHandled(vm))
            m_outstandingRejectedPromises.set(&promise, &promise);
    }
}

void RejectedPromiseTracker::reportRejectionHandled(Ref<DOMPromise>&& rejectedPromise)
{
    // https://html.spec.whatwg.org/multipage/webappapis.html#the-hostpromiserejectiontracker-implementation

    Ref vm = m_context->vm();
    JSC::JSLockHolder lock(vm);

    if (rejectedPromise->isSuspended())
        return;

    auto& promise = *rejectedPromise->promise();

    PromiseRejectionEvent::Init initializer;
    initializer.promise = rejectedPromise.ptr();
    initializer.reason = promise.result(vm);

    Ref event = PromiseRejectionEvent::create(eventNames().rejectionhandledEvent, initializer);
    RefPtr target = m_context->errorEventTarget();
    target->dispatchEvent(event);
}

} // namespace WebCore
