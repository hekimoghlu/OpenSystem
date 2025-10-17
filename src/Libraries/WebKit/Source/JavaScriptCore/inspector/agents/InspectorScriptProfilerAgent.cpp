/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include "InspectorScriptProfilerAgent.h"

#include "Debugger.h"
#include "DeferGCInlines.h"
#include "HeapInlines.h"
#include "InspectorEnvironment.h"
#include "SamplingProfiler.h"
#include <wtf/Stopwatch.h>
#include <wtf/TZoneMallocInlines.h>

namespace Inspector {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(InspectorScriptProfilerAgent);

InspectorScriptProfilerAgent::InspectorScriptProfilerAgent(AgentContext& context)
    : InspectorAgentBase("ScriptProfiler"_s)
    , m_frontendDispatcher(makeUnique<ScriptProfilerFrontendDispatcher>(context.frontendRouter))
    , m_backendDispatcher(ScriptProfilerBackendDispatcher::create(context.backendDispatcher, this))
    , m_environment(context.environment)
{
}

InspectorScriptProfilerAgent::~InspectorScriptProfilerAgent() = default;

void InspectorScriptProfilerAgent::didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*)
{
}

void InspectorScriptProfilerAgent::willDestroyFrontendAndBackend(DisconnectReason)
{
    // Stop tracking without sending results.
    if (m_tracking) {
        m_tracking = false;
        m_activeEvaluateScript = false;
        m_environment.debugger()->setProfilingClient(nullptr);

        // Stop sampling without processing the samples.
        stopSamplingWhenDisconnecting();
    }
}

Protocol::ErrorStringOr<void> InspectorScriptProfilerAgent::startTracking(std::optional<bool>&& includeSamples)
{
    if (m_tracking)
        return { };

    m_tracking = true;

    auto& stopwatch = m_environment.executionStopwatch();

#if ENABLE(SAMPLING_PROFILER)
    if (includeSamples && *includeSamples) {
        VM& vm = m_environment.debugger()->vm();
        SamplingProfiler& samplingProfiler = vm.ensureSamplingProfiler(stopwatch);

        Locker locker { samplingProfiler.getLock() };
        samplingProfiler.setStopWatch(stopwatch);
        samplingProfiler.noticeCurrentThreadAsJSCExecutionThreadWithLock();
        samplingProfiler.startWithLock();
        m_enabledSamplingProfiler = true;
    }
#else
    UNUSED_PARAM(includeSamples);
#endif // ENABLE(SAMPLING_PROFILER)

    m_environment.debugger()->setProfilingClient(this);

    m_frontendDispatcher->trackingStart(stopwatch.elapsedTime().seconds());

    return { };
}

Protocol::ErrorStringOr<void> InspectorScriptProfilerAgent::stopTracking()
{
    if (!m_tracking)
        return { };

    m_tracking = false;
    m_activeEvaluateScript = false;

    m_environment.debugger()->setProfilingClient(nullptr);

    trackingComplete();

    return { };
}

bool InspectorScriptProfilerAgent::isAlreadyProfiling() const
{
    return m_activeEvaluateScript;
}

Seconds InspectorScriptProfilerAgent::willEvaluateScript()
{
    m_activeEvaluateScript = true;

#if ENABLE(SAMPLING_PROFILER)
    if (m_enabledSamplingProfiler) {
        SamplingProfiler* samplingProfiler = m_environment.debugger()->vm().samplingProfiler();
        RELEASE_ASSERT(samplingProfiler);
        samplingProfiler->noticeCurrentThreadAsJSCExecutionThread();
    }
#endif

    return m_environment.executionStopwatch().elapsedTime();
}

void InspectorScriptProfilerAgent::didEvaluateScript(Seconds startTime, ProfilingReason reason)
{
    m_activeEvaluateScript = false;

    Seconds endTime = m_environment.executionStopwatch().elapsedTime();

    addEvent(startTime, endTime, reason);
}

static Protocol::ScriptProfiler::EventType toProtocol(ProfilingReason reason)
{
    switch (reason) {
    case ProfilingReason::API:
        return Protocol::ScriptProfiler::EventType::API;
    case ProfilingReason::Microtask:
        return Protocol::ScriptProfiler::EventType::Microtask;
    case ProfilingReason::Other:
        return Protocol::ScriptProfiler::EventType::Other;
    }

    ASSERT_NOT_REACHED();
    return Protocol::ScriptProfiler::EventType::Other;
}

void InspectorScriptProfilerAgent::addEvent(Seconds startTime, Seconds endTime, ProfilingReason reason)
{
    ASSERT(endTime >= startTime);

    auto event = Protocol::ScriptProfiler::Event::create()
        .setStartTime(startTime.seconds())
        .setEndTime(endTime.seconds())
        .setType(toProtocol(reason))
        .release();

    m_frontendDispatcher->trackingUpdate(WTFMove(event));
}

#if ENABLE(SAMPLING_PROFILER)
static Ref<Protocol::ScriptProfiler::Samples> buildSamples(VM& vm, Vector<SamplingProfiler::StackTrace>&& samplingProfilerStackTraces)
{
    auto stackTraces = JSON::ArrayOf<Protocol::ScriptProfiler::StackTrace>::create();
    for (SamplingProfiler::StackTrace& stackTrace : samplingProfilerStackTraces) {
        auto frames = JSON::ArrayOf<Protocol::ScriptProfiler::StackFrame>::create();
        for (SamplingProfiler::StackFrame& stackFrame : stackTrace.frames) {
            auto frameObject = Protocol::ScriptProfiler::StackFrame::create()
                .setSourceID(String::number(std::get<1>(stackFrame.sourceProviderAndID())))
                .setName(stackFrame.displayName(vm))
                .setLine(stackFrame.functionStartLine())
                .setColumn(stackFrame.functionStartColumn())
                .setUrl(stackFrame.url())
                .release();

            if (stackFrame.hasExpressionInfo()) {
                Ref<Protocol::ScriptProfiler::ExpressionLocation> expressionLocation = Protocol::ScriptProfiler::ExpressionLocation::create()
                    .setLine(stackFrame.lineNumber())
                    .setColumn(stackFrame.columnNumber())
                    .release();
                frameObject->setExpressionLocation(WTFMove(expressionLocation));
            }

            frames->addItem(WTFMove(frameObject));
        }
        Ref<Protocol::ScriptProfiler::StackTrace> inspectorStackTrace = Protocol::ScriptProfiler::StackTrace::create()
            .setTimestamp(stackTrace.stopwatchTimestamp.seconds())
            .setStackFrames(WTFMove(frames))
            .release();
        stackTraces->addItem(WTFMove(inspectorStackTrace));
    }

    return Protocol::ScriptProfiler::Samples::create()
        .setStackTraces(WTFMove(stackTraces))
        .release();
}
#endif // ENABLE(SAMPLING_PROFILER)

void InspectorScriptProfilerAgent::trackingComplete()
{
    auto stopwatchTimestamp = m_environment.executionStopwatch().elapsedTime().seconds();

#if ENABLE(SAMPLING_PROFILER)
    if (m_enabledSamplingProfiler) {
        VM& vm = m_environment.debugger()->vm();
        JSLockHolder lock(vm);
        DeferGC deferGC(vm); // This is required because we will have raw pointers into the heap after we releaseStackTraces().
        SamplingProfiler* samplingProfiler = vm.samplingProfiler();
        RELEASE_ASSERT(samplingProfiler);

        Locker locker { samplingProfiler->getLock() };
        samplingProfiler->pause();
        Vector<SamplingProfiler::StackTrace> stackTraces = samplingProfiler->releaseStackTraces();
        locker.unlockEarly();

        Ref<Protocol::ScriptProfiler::Samples> samples = buildSamples(vm, WTFMove(stackTraces));

        m_enabledSamplingProfiler = false;

        m_frontendDispatcher->trackingComplete(stopwatchTimestamp, WTFMove(samples));
    } else
        m_frontendDispatcher->trackingComplete(stopwatchTimestamp, nullptr);
#else
    m_frontendDispatcher->trackingComplete(stopwatchTimestamp, nullptr);
#endif // ENABLE(SAMPLING_PROFILER)
}

void InspectorScriptProfilerAgent::stopSamplingWhenDisconnecting()
{
#if ENABLE(SAMPLING_PROFILER)
    if (!m_enabledSamplingProfiler)
        return;

    VM& vm = m_environment.debugger()->vm();
    JSLockHolder lock(vm);
    SamplingProfiler* samplingProfiler = vm.samplingProfiler();
    RELEASE_ASSERT(samplingProfiler);
    Locker locker { samplingProfiler->getLock() };
    samplingProfiler->pause();
    samplingProfiler->clearData();

    m_enabledSamplingProfiler = false;
#endif
}

} // namespace Inspector
