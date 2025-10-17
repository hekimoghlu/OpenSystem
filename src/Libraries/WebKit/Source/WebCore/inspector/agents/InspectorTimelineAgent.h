/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#pragma once

#include "InspectorWebAgentBase.h"
#include "LayoutRect.h"
#include <JavaScriptCore/Debugger.h>
#include <JavaScriptCore/DebuggerPrimitives.h>
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <wtf/JSONValues.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class Event;
class FloatQuad;
class Frame;
class LocalFrame;
class RenderObject;
class RunLoopObserver;

enum class TimelineRecordType {
    EventDispatch,
    ScheduleStyleRecalculation,
    RecalculateStyles,
    InvalidateLayout,
    Layout,
    Paint,
    Composite,
    RenderingFrame,

    TimerInstall,
    TimerRemove,
    TimerFire,

    EvaluateScript,

    TimeStamp,
    Time,
    TimeEnd,

    FunctionCall,
    ProbeSample,
    ConsoleProfile,

    RequestAnimationFrame,
    CancelAnimationFrame,
    FireAnimationFrame,
    
    ObserverCallback,

    Screenshot,
};

class InspectorTimelineAgent final : public InspectorAgentBase , public Inspector::TimelineBackendDispatcherHandler , public JSC::Debugger::Observer {
    WTF_MAKE_NONCOPYABLE(InspectorTimelineAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorTimelineAgent);
public:
    InspectorTimelineAgent(PageAgentContext&);
    ~InspectorTimelineAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // TimelineBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<void> start(std::optional<int>&& maxCallStackDepth);
    Inspector::Protocol::ErrorStringOr<void> stop();
    Inspector::Protocol::ErrorStringOr<void> setAutoCaptureEnabled(bool);
    Inspector::Protocol::ErrorStringOr<void> setInstruments(Ref<JSON::Array>&&);

    // JSC::Debugger::Observer
    void breakpointActionProbe(JSC::JSGlobalObject*, JSC::BreakpointActionID, unsigned batchId, unsigned sampleId, JSC::JSValue result);

    // InspectorInstrumentation
    void didInstallTimer(int timerId, Seconds timeout, bool singleShot, LocalFrame*);
    void didRemoveTimer(int timerId, LocalFrame*);
    void willFireTimer(int timerId, LocalFrame*);
    void didFireTimer();
    void willCallFunction(const String& scriptName, int scriptLine, int scriptColumn, LocalFrame*);
    void didCallFunction(LocalFrame*);
    void willDispatchEvent(const Event&, LocalFrame*);
    void didDispatchEvent(bool defaultPrevented);
    void willEvaluateScript(const String&, int lineNumber, int columnNumber, LocalFrame&);
    void didEvaluateScript(LocalFrame&);
    void didInvalidateLayout(LocalFrame&);
    void willLayout(LocalFrame&);
    void didLayout(RenderObject&);
    void willComposite(LocalFrame&);
    void didComposite();
    void willPaint(LocalFrame&);
    void didPaint(RenderObject&, const LayoutRect&);
    void willRecalculateStyle(LocalFrame*);
    void didRecalculateStyle();
    void didScheduleStyleRecalculation(LocalFrame*);
    void didTimeStamp(Frame&, const String&);
    void didPerformanceMark(const String&, std::optional<MonotonicTime>, Frame*);
    void didRequestAnimationFrame(int callbackId, LocalFrame*);
    void didCancelAnimationFrame(int callbackId, LocalFrame*);
    void willFireAnimationFrame(int callbackId, LocalFrame*);
    void didFireAnimationFrame();
    void willFireObserverCallback(const String& callbackType, LocalFrame*);
    void didFireObserverCallback();
    void time(Frame&, const String&);
    void timeEnd(Frame&, const String&);
    void mainFrameStartedLoading();
    void mainFrameNavigated();
    void didCompleteRenderingFrame();

    // Console
    void startFromConsole(JSC::JSGlobalObject*, const String& title);
    void stopFromConsole(JSC::JSGlobalObject*, const String& title);

private:
    void startProgrammaticCapture();
    void stopProgrammaticCapture();

    enum class InstrumentState { Start, Stop };
    void toggleInstruments(InstrumentState);
    void toggleScriptProfilerInstrument(InstrumentState);
    void toggleHeapInstrument(InstrumentState);
    void toggleCPUInstrument(InstrumentState);
    void toggleMemoryInstrument(InstrumentState);
    void toggleTimelineInstrument(InstrumentState);
    void toggleAnimationInstrument(InstrumentState);
    void disableBreakpoints();
    void enableBreakpoints();

    void captureScreenshot();

    friend class TimelineRecordStack;

    struct TimelineRecordEntry {
        TimelineRecordEntry(Ref<JSON::Object>&& record, Ref<JSON::Object>&& data, RefPtr<JSON::Array>&& children, TimelineRecordType type)
            : record(WTFMove(record))
            , data(WTFMove(data))
            , children(WTFMove(children))
            , type(type)
        {
        }

        Ref<JSON::Object> record;
        Ref<JSON::Object> data;
        RefPtr<JSON::Array> children;
        TimelineRecordType type;
    };

    void internalStart(std::optional<int>&& maxCallStackDepth);
    void internalStop();
    double timestamp();
    std::optional<double> timestampFromMonotonicTime(MonotonicTime);

    void sendEvent(Ref<JSON::Object>&&);
    void appendRecord(Ref<JSON::Object>&& data, TimelineRecordType, bool captureCallStack, Frame*, std::optional<double> startTime = std::nullopt);
    void pushCurrentRecord(Ref<JSON::Object>&&, TimelineRecordType, bool captureCallStack, LocalFrame*, std::optional<double> startTime = std::nullopt);
    void pushCurrentRecord(const TimelineRecordEntry& record) { m_recordStack.append(record); }

    TimelineRecordEntry createRecordEntry(Ref<JSON::Object>&& data, TimelineRecordType, bool captureCallStack, LocalFrame*, std::optional<double> startTime = std::nullopt);

    void setFrameIdentifier(JSON::Object* record, Frame*);

    void didCompleteRecordEntry(const TimelineRecordEntry&);
    void didCompleteCurrentRecord(TimelineRecordType);

    void addRecordToTimeline(Ref<JSON::Object>&&, TimelineRecordType);

    std::unique_ptr<Inspector::TimelineFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::TimelineBackendDispatcher> m_backendDispatcher;
    WeakRef<Page> m_inspectedPage;

    Vector<TimelineRecordEntry> m_recordStack;
    Vector<TimelineRecordEntry> m_pendingConsoleProfileRecords;

    int m_maxCallStackDepth { 5 };

    bool m_tracking { false };
    bool m_trackingFromFrontend { false };
    bool m_programmaticCaptureRestoreBreakpointActiveValue { false };

    bool m_autoCaptureEnabled { false };
    enum class AutoCapturePhase { None, BeforeLoad, FirstNavigation, AfterFirstNavigation };
    AutoCapturePhase m_autoCapturePhase { AutoCapturePhase::None };
    Vector<Inspector::Protocol::Timeline::Instrument> m_instruments;

#if PLATFORM(COCOA)
    std::unique_ptr<WebCore::RunLoopObserver> m_frameStartObserver;
    std::unique_ptr<WebCore::RunLoopObserver> m_frameStopObserver;
    int m_runLoopNestingLevel { 0 };
#elif USE(GLIB_EVENT_LOOP)
    std::unique_ptr<RunLoop::Observer> m_runLoopObserver;
#endif
    bool m_startedComposite { false };
    bool m_isCapturingScreenshot { false };
};

} // namespace WebCore
