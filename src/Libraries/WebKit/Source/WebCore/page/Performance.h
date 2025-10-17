/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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

#include "ContextDestructionObserver.h"
#include "DOMHighResTimeStamp.h"
#include "EventTarget.h"
#include "ExceptionOr.h"
#include "ReducedResolutionSeconds.h"
#include "ScriptExecutionContext.h"
#include "Timer.h"
#include <variant>
#include <wtf/ListHashSet.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

class CachedResource;
class Document;
class DocumentLoadTiming;
class DocumentLoader;
class NetworkLoadMetrics;
class PerformanceUserTiming;
class PerformanceEntry;
class PerformanceMark;
class PerformanceMeasure;
class PerformanceNavigation;
class PerformanceNavigationTiming;
class PerformanceObserver;
class PerformancePaintTiming;
class PerformanceTiming;
class ResourceResponse;
class ResourceTiming;
class ScriptExecutionContext;
struct PerformanceMarkOptions;
struct PerformanceMeasureOptions;

class Performance final : public RefCounted<Performance>, public ContextDestructionObserver, public EventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(Performance);
public:
    static Ref<Performance> create(ScriptExecutionContext* context, MonotonicTime timeOrigin) { return adoptRef(*new Performance(context, timeOrigin)); }
    ~Performance();

    DOMHighResTimeStamp now() const;
    DOMHighResTimeStamp timeOrigin() const;
    ReducedResolutionSeconds nowInReducedResolutionSeconds() const;

    PerformanceNavigation* navigation();
    PerformanceTiming* timing();

    Vector<Ref<PerformanceEntry>> getEntries() const;
    Vector<Ref<PerformanceEntry>> getEntriesByType(const String& entryType) const;
    Vector<Ref<PerformanceEntry>> getEntriesByName(const String& name, const String& entryType) const;
    void appendBufferedEntriesByType(const String& entryType, Vector<Ref<PerformanceEntry>>&, PerformanceObserver&) const;

    void clearResourceTimings();
    void setResourceTimingBufferSize(unsigned);

    ExceptionOr<Ref<PerformanceMark>> mark(JSC::JSGlobalObject&, const String& markName, std::optional<PerformanceMarkOptions>&&);
    void clearMarks(const String& markName);

    using StartOrMeasureOptions = std::variant<String, PerformanceMeasureOptions>;
    ExceptionOr<Ref<PerformanceMeasure>> measure(JSC::JSGlobalObject&, const String& measureName, std::optional<StartOrMeasureOptions>&&, const String& endMark);
    void clearMeasures(const String& measureName);

    void addNavigationTiming(DocumentLoader&, Document&, CachedResource&, const DocumentLoadTiming&, const NetworkLoadMetrics&);
    void navigationFinished(const NetworkLoadMetrics&);
    void addResourceTiming(ResourceTiming&&);

    void reportFirstContentfulPaint();

    void removeAllObservers();
    void registerPerformanceObserver(PerformanceObserver&);
    void unregisterPerformanceObserver(PerformanceObserver&);

    static void allowHighPrecisionTime();
    static Seconds timeResolution();
    static Seconds reduceTimeResolution(Seconds);

    DOMHighResTimeStamp relativeTimeFromTimeOriginInReducedResolution(MonotonicTime) const;
    MonotonicTime monotonicTimeFromRelativeTime(DOMHighResTimeStamp) const;

    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

    using RefCounted::ref;
    using RefCounted::deref;

    void scheduleNavigationObservationTaskIfNeeded();

    PerformanceNavigationTiming* navigationTiming() { return m_navigationTiming.get(); }

private:
    Performance(ScriptExecutionContext*, MonotonicTime timeOrigin);

    void contextDestroyed() override;

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::Performance; }

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    bool isResourceTimingBufferFull() const;
    void resourceTimingBufferFullTimerFired();

    void queueEntry(PerformanceEntry&);
    void scheduleTaskIfNeeded();

    mutable RefPtr<PerformanceNavigation> m_navigation;
    mutable RefPtr<PerformanceTiming> m_timing;

    // https://w3c.github.io/resource-timing/#sec-extensions-performance-interface recommends initial buffer size of 250.
    Vector<Ref<PerformanceEntry>> m_resourceTimingBuffer;
    unsigned m_resourceTimingBufferSize { 250 };

    Timer m_resourceTimingBufferFullTimer;
    Vector<Ref<PerformanceEntry>> m_backupResourceTimingBuffer;

    // https://w3c.github.io/resource-timing/#dfn-resource-timing-buffer-full-flag
    bool m_resourceTimingBufferFullFlag { false };
    bool m_waitingForBackupBufferToBeProcessed { false };
    bool m_hasScheduledTimingBufferDeliveryTask { false };

    MonotonicTime m_timeOrigin;

    RefPtr<PerformanceNavigationTiming> m_navigationTiming;
    RefPtr<PerformancePaintTiming> m_firstContentfulPaint;
    std::unique_ptr<PerformanceUserTiming> m_userTiming;

    ListHashSet<RefPtr<PerformanceObserver>> m_observers;
};

}
