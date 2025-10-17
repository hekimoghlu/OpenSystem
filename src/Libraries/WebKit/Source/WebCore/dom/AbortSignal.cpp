/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#include "AbortSignal.h"

#include "AbortAlgorithm.h"
#include "DOMException.h"
#include "DOMTimer.h"
#include "Event.h"
#include "EventNames.h"
#include "JSDOMException.h"
#include "ScriptExecutionContext.h"
#include "WebCoreOpaqueRoot.h"
#include <JavaScriptCore/Exception.h>
#include <JavaScriptCore/JSCast.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(AbortSignal);

Ref<AbortSignal> AbortSignal::create(ScriptExecutionContext* context)
{
    return adoptRef(*new AbortSignal(context));
}

// https://dom.spec.whatwg.org/#dom-abortsignal-abort
Ref<AbortSignal> AbortSignal::abort(JSDOMGlobalObject& globalObject, ScriptExecutionContext& context, JSC::JSValue reason)
{
    ASSERT(reason);
    if (reason.isUndefined())
        reason = toJS(&globalObject, &globalObject, DOMException::create(ExceptionCode::AbortError));
    return adoptRef(*new AbortSignal(&context, Aborted::Yes, reason));
}

// https://dom.spec.whatwg.org/#dom-abortsignal-timeout
Ref<AbortSignal> AbortSignal::timeout(ScriptExecutionContext& context, uint64_t milliseconds)
{
    Ref signal = AbortSignal::create(&context);
    signal->setHasActiveTimeoutTimer(true);
    auto action = [signal](ScriptExecutionContext& context) mutable {
        signal->setHasActiveTimeoutTimer(false);

        auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(context.globalObject());
        if (!globalObject)
            return;

        Locker locker { globalObject->vm().apiLock() };
        signal->signalAbort(toJS(globalObject, globalObject, DOMException::create(ExceptionCode::TimeoutError)));
    };
    DOMTimer::install(context, WTFMove(action), Seconds::fromMilliseconds(milliseconds), DOMTimer::Type::SingleShot);
    return signal;
}

Ref<AbortSignal> AbortSignal::any(ScriptExecutionContext& context, const Vector<Ref<AbortSignal>>& signals)
{
    Ref resultSignal = AbortSignal::create(&context);

    auto abortedSignalIndex = signals.findIf([](auto& signal) {
        return signal->aborted();
    });
    if (abortedSignalIndex != notFound) {
        resultSignal->signalAbort(signals[abortedSignalIndex]->reason().getValue());
        return resultSignal;
    }

    resultSignal->markAsDependent();
    for (auto& signal : signals)
        resultSignal->addSourceSignal(signal);

    return resultSignal;
}

AbortSignal::AbortSignal(ScriptExecutionContext* context, Aborted aborted, JSC::JSValue reason)
    : ContextDestructionObserver(context)
    , m_reason(reason)
    , m_aborted(aborted == Aborted::Yes)
{
    ASSERT(reason);
}

AbortSignal::~AbortSignal() = default;

void AbortSignal::addSourceSignal(AbortSignal& signal)
{
    if (signal.isDependent()) {
        for (Ref sourceSignal : signal.sourceSignals())
            addSourceSignal(sourceSignal);
        return;
    }
    ASSERT(!signal.aborted());
    ASSERT(signal.sourceSignals().isEmptyIgnoringNullReferences());
    m_sourceSignals.add(signal);
    signal.addDependentSignal(*this);
}

void AbortSignal::addDependentSignal(AbortSignal& signal)
{
    m_dependentSignals.add(signal);
}

// https://dom.spec.whatwg.org/#abortsignal-signal-abort
void AbortSignal::signalAbort(JSC::JSValue reason)
{
    // 1. If signal's aborted flag is set, then return.
    if (m_aborted)
        return;

    // 2. ... if the reason is not given, set it to a new "AbortError" DOMException.
    ASSERT(reason);
    if (reason.isUndefined()) {
        auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(protectedScriptExecutionContext()->globalObject());
        if (!globalObject)
            return;
        reason = toJS(globalObject, globalObject, DOMException::create(ExceptionCode::AbortError));
    }

    // 2. Set signalâ€™s abort reason to reason if it is given; otherwise to a new "AbortError" DOMException.
    markAborted(reason);

    Vector<Ref<AbortSignal>> dependentSignalsToAbort;

    for (Ref dependentSignal : std::exchange(m_dependentSignals, { })) {
        if (!dependentSignal->aborted()) {
            dependentSignal->markAborted(reason);
            dependentSignalsToAbort.append(WTFMove(dependentSignal));
        }
    }

    // 5. Run the abort steps
    runAbortSteps();

    // 6. For each dependentSignal of dependentSignalsToAbort, run the abort steps for dependentSignal.
    for (auto& dependentSignal : dependentSignalsToAbort)
        dependentSignal->runAbortSteps();
}

void AbortSignal::markAborted(JSC::JSValue reason)
{
    m_aborted = true;
    m_sourceSignals.clear();

    // FIXME: This code is wrong: we should emit a write-barrier. Otherwise, GC can collect it.
    // https://bugs.webkit.org/show_bug.cgi?id=236353
    ASSERT(reason);
    m_reason.setWeakly(reason);
}

void AbortSignal::runAbortSteps()
{
    auto reason = m_reason.getValue();
    ASSERT(reason);

    // 1. For each algorithm of signal's abort algorithms: run algorithm.
    //    2. Empty signal's abort algorithms. (std::exchange empties)
    for (auto& algorithm : std::exchange(m_algorithms, { }))
        algorithm.second(reason);

    // 3. Fire an event named abort at signal.
    dispatchEvent(Event::create(eventNames().abortEvent, Event::CanBubble::No, Event::IsCancelable::No));
}

// https://dom.spec.whatwg.org/#abortsignal-follow
void AbortSignal::signalFollow(AbortSignal& signal)
{
    if (aborted())
        return;

    if (signal.aborted()) {
        signalAbort(signal.reason().getValue());
        return;
    }

    ASSERT(!m_followingSignal);
    m_followingSignal = signal;
    signal.addAlgorithm([weakThis = WeakPtr { *this }](JSC::JSValue reason) {
        if (RefPtr signal = weakThis.get())
            signal->signalAbort(reason);
    });
}

void AbortSignal::eventListenersDidChange()
{
    m_hasAbortEventListener = hasEventListeners(eventNames().abortEvent);
}

uint32_t AbortSignal::addAbortAlgorithmToSignal(AbortSignal& signal, Ref<AbortAlgorithm>&& algorithm)
{
    if (signal.aborted()) {
        algorithm->handleEvent(signal.m_reason.getValue());
        return 0;
    }
    return signal.addAlgorithm([algorithm = WTFMove(algorithm)](JSC::JSValue value) mutable {
        algorithm->handleEvent(value);
    });
}

void AbortSignal::removeAbortAlgorithmFromSignal(AbortSignal& signal, uint32_t algorithmIdentifier)
{
    signal.removeAlgorithm(algorithmIdentifier);
}

uint32_t AbortSignal::addAlgorithm(Algorithm&& algorithm)
{
    m_algorithms.append(std::make_pair(++m_algorithmIdentifier, WTFMove(algorithm)));
    return m_algorithmIdentifier;
}

void AbortSignal::removeAlgorithm(uint32_t algorithmIdentifier)
{
    m_algorithms.removeFirstMatching([algorithmIdentifier](auto& pair) {
        return pair.first == algorithmIdentifier;
    });
}

void AbortSignal::throwIfAborted(JSC::JSGlobalObject& lexicalGlobalObject)
{
    if (!aborted())
        return;

    Ref vm = lexicalGlobalObject.vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    throwException(&lexicalGlobalObject, scope, m_reason.getValue());
}

WebCoreOpaqueRoot root(AbortSignal* signal)
{
    return WebCoreOpaqueRoot { signal };
}

} // namespace WebCore
