/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#include "Performance.h"

#include "Document.h"
#include "DocumentLoader.h"
#include "Event.h"
#include "EventLoop.h"
#include "EventNames.h"
#include "LocalFrame.h"
#include "PerformanceEntry.h"
#include "PerformanceMarkOptions.h"
#include "PerformanceMeasureOptions.h"
#include "PerformanceNavigation.h"
#include "PerformanceNavigationTiming.h"
#include "PerformanceObserver.h"
#include "PerformancePaintTiming.h"
#include "PerformanceResourceTiming.h"
#include "PerformanceTiming.h"
#include "PerformanceUserTiming.h"
#include "ResourceResponse.h"
#include "ScriptExecutionContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Performance);

static Seconds timePrecision { 1_ms };

Performance::Performance(ScriptExecutionContext* context, MonotonicTime timeOrigin)
    : ContextDestructionObserver(context)
    , m_resourceTimingBufferFullTimer(*this, &Performance::resourceTimingBufferFullTimerFired) // FIXME: Migrate this to the event loop as well. https://bugs.webkit.org/show_bug.cgi?id=229044
    , m_timeOrigin(timeOrigin)
{
    ASSERT(m_timeOrigin);
}

Performance::~Performance() = default;

void Performance::contextDestroyed()
{
    m_resourceTimingBufferFullTimer.stop();
    ContextDestructionObserver::contextDestroyed();
}

DOMHighResTimeStamp Performance::now() const
{
    return nowInReducedResolutionSeconds().milliseconds();
}

DOMHighResTimeStamp Performance::timeOrigin() const
{
    return reduceTimeResolution(m_timeOrigin.approximateWallTime().secondsSinceEpoch()).milliseconds();
}

ReducedResolutionSeconds Performance::nowInReducedResolutionSeconds() const
{
    Seconds now = MonotonicTime::now() - m_timeOrigin;
    return reduceTimeResolution(now);
}

Seconds Performance::reduceTimeResolution(Seconds seconds)
{
    return seconds.reduceTimeResolution(timePrecision);
}

void Performance::allowHighPrecisionTime()
{
    timePrecision = Seconds::highTimePrecision();
}

Seconds Performance::timeResolution()
{
    return timePrecision;
}

DOMHighResTimeStamp Performance::relativeTimeFromTimeOriginInReducedResolution(MonotonicTime timestamp) const
{
    Seconds seconds = timestamp - m_timeOrigin;
    return reduceTimeResolution(seconds).milliseconds();
}

MonotonicTime Performance::monotonicTimeFromRelativeTime(DOMHighResTimeStamp relativeTime) const
{
    return m_timeOrigin + Seconds::fromMilliseconds(relativeTime);
}

PerformanceNavigation* Performance::navigation()
{
    if (!is<Document>(scriptExecutionContext()))
        return nullptr;

    ASSERT(isMainThread());
    if (!m_navigation)
        m_navigation = PerformanceNavigation::create(downcast<Document>(*scriptExecutionContext()).domWindow());
    return m_navigation.get();
}

PerformanceTiming* Performance::timing()
{
    if (!is<Document>(scriptExecutionContext()))
        return nullptr;

    ASSERT(isMainThread());
    if (!m_timing)
        m_timing = PerformanceTiming::create(downcast<Document>(*scriptExecutionContext()).domWindow());
    return m_timing.get();
}

Vector<Ref<PerformanceEntry>> Performance::getEntries() const
{
    Vector<Ref<PerformanceEntry>> entries;

    if (m_navigationTiming)
        entries.append(*m_navigationTiming);

    entries.appendVector(m_resourceTimingBuffer);

    if (m_userTiming) {
        entries.appendVector(m_userTiming->getMarks());
        entries.appendVector(m_userTiming->getMeasures());
    }

    if (m_firstContentfulPaint)
        entries.append(*m_firstContentfulPaint);

    std::sort(entries.begin(), entries.end(), PerformanceEntry::startTimeCompareLessThan);
    return entries;
}

Vector<Ref<PerformanceEntry>> Performance::getEntriesByType(const String& entryType) const
{
    Vector<Ref<PerformanceEntry>> entries;

    if (m_navigationTiming && entryType == "navigation"_s)
        entries.append(*m_navigationTiming);
    
    if (entryType == "resource"_s)
        entries.appendVector(m_resourceTimingBuffer);

    if (m_firstContentfulPaint && entryType == "paint"_s)
        entries.append(*m_firstContentfulPaint);

    if (m_userTiming) {
        if (entryType == "mark"_s)
            entries.appendVector(m_userTiming->getMarks());
        else if (entryType == "measure"_s)
            entries.appendVector(m_userTiming->getMeasures());
    }

    std::sort(entries.begin(), entries.end(), PerformanceEntry::startTimeCompareLessThan);
    return entries;
}

Vector<Ref<PerformanceEntry>> Performance::getEntriesByName(const String& name, const String& entryType) const
{
    Vector<Ref<PerformanceEntry>> entries;

    if (m_navigationTiming && (entryType.isNull() || entryType == "navigation"_s) && name == m_navigationTiming->name())
        entries.append(*m_navigationTiming);

    if (entryType.isNull() || entryType == "resource"_s) {
        for (auto& resource : m_resourceTimingBuffer) {
            if (resource->name() == name)
                entries.append(resource);
        }
    }

    if (m_firstContentfulPaint && (entryType.isNull() || entryType == "paint"_s) && name == "first-contentful-paint"_s)
        entries.append(*m_firstContentfulPaint);

    if (m_userTiming) {
        if (entryType.isNull() || entryType == "mark"_s)
            entries.appendVector(m_userTiming->getMarks(name));
        if (entryType.isNull() || entryType == "measure"_s)
            entries.appendVector(m_userTiming->getMeasures(name));
    }

    std::sort(entries.begin(), entries.end(), PerformanceEntry::startTimeCompareLessThan);
    return entries;
}

void Performance::appendBufferedEntriesByType(const String& entryType, Vector<Ref<PerformanceEntry>>& entries, PerformanceObserver& observer) const
{
    if (m_navigationTiming
        && entryType == "navigation"_s
        && !observer.hasNavigationTiming()) {
        entries.append(*m_navigationTiming);
        observer.addedNavigationTiming();
    }

    if (entryType == "resource"_s)
        entries.appendVector(m_resourceTimingBuffer);

    if (entryType == "paint"_s && m_firstContentfulPaint)
        entries.append(*m_firstContentfulPaint);

    if (m_userTiming) {
        if (entryType.isNull() || entryType == "mark"_s)
            entries.appendVector(m_userTiming->getMarks());
        if (entryType.isNull() || entryType == "measure"_s)
            entries.appendVector(m_userTiming->getMeasures());
    }
}

void Performance::clearResourceTimings()
{
    m_resourceTimingBuffer.clear();
    m_resourceTimingBufferFullFlag = false;
}

void Performance::setResourceTimingBufferSize(unsigned size)
{
    m_resourceTimingBufferSize = size;
    m_resourceTimingBufferFullFlag = false;
}

void Performance::reportFirstContentfulPaint()
{
    ASSERT(!m_firstContentfulPaint);
    m_firstContentfulPaint = PerformancePaintTiming::createFirstContentfulPaint(now());
    queueEntry(*m_firstContentfulPaint);
}

void Performance::addNavigationTiming(DocumentLoader& documentLoader, Document& document, CachedResource& resource, const DocumentLoadTiming& timing, const NetworkLoadMetrics& metrics)
{
    m_navigationTiming = PerformanceNavigationTiming::create(m_timeOrigin, resource, timing, metrics, document.eventTiming(), document.securityOrigin(), documentLoader.triggeringAction().type());
}

void Performance::navigationFinished(const NetworkLoadMetrics& metrics)
{
    if (!m_navigationTiming)
        return;
    m_navigationTiming->navigationFinished(metrics);

    queueEntry(*m_navigationTiming);
}

void Performance::addResourceTiming(ResourceTiming&& resourceTiming)
{
    ASSERT(scriptExecutionContext());

    auto entry = PerformanceResourceTiming::create(m_timeOrigin, WTFMove(resourceTiming));

    if (m_waitingForBackupBufferToBeProcessed) {
        m_backupResourceTimingBuffer.append(WTFMove(entry));
        return;
    }

    if (m_resourceTimingBufferFullFlag) {
        // We fired resourcetimingbufferfull event but the author script didn't clear the buffer.
        // Notify performance observers but don't add it to the buffer.
        queueEntry(entry.get());
        return;
    }

    if (isResourceTimingBufferFull()) {
        ASSERT(!m_resourceTimingBufferFullTimer.isActive());
        m_backupResourceTimingBuffer.append(WTFMove(entry));
        m_waitingForBackupBufferToBeProcessed = true;
        m_resourceTimingBufferFullTimer.startOneShot(0_s);
        return;
    }

    queueEntry(entry.get());
    m_resourceTimingBuffer.append(WTFMove(entry));
}

bool Performance::isResourceTimingBufferFull() const
{
    return m_resourceTimingBuffer.size() >= m_resourceTimingBufferSize;
}

void Performance::resourceTimingBufferFullTimerFired()
{
    ASSERT(scriptExecutionContext());

    while (!m_backupResourceTimingBuffer.isEmpty()) {
        auto beforeCount = m_backupResourceTimingBuffer.size();

        auto backupBuffer = std::exchange(m_backupResourceTimingBuffer, { });
        ASSERT(m_backupResourceTimingBuffer.isEmpty());

        if (isResourceTimingBufferFull()) {
            m_resourceTimingBufferFullFlag = true;
            dispatchEvent(Event::create(eventNames().resourcetimingbufferfullEvent, Event::CanBubble::No, Event::IsCancelable::No));
        }

        if (m_resourceTimingBufferFullFlag) {
            for (auto& entry : backupBuffer)
                queueEntry(entry);
            // Dispatching resourcetimingbufferfull event may have inserted more entries.
            for (auto& entry : std::exchange(m_backupResourceTimingBuffer, { }))
                queueEntry(entry);
            break;
        }

        // More entries may have added while dispatching resourcetimingbufferfull event.
        backupBuffer.appendVector(std::exchange(m_backupResourceTimingBuffer, { }));

        for (auto& entry : backupBuffer) {
            if (!isResourceTimingBufferFull()) {
                m_resourceTimingBuffer.append(entry.copyRef());
                queueEntry(entry);
            } else
                m_backupResourceTimingBuffer.append(entry.copyRef());
        }

        auto afterCount = m_backupResourceTimingBuffer.size();

        if (beforeCount <= afterCount) {
            m_backupResourceTimingBuffer.clear();
            break;
        }
    }
    m_waitingForBackupBufferToBeProcessed = false;
}

ExceptionOr<Ref<PerformanceMark>> Performance::mark(JSC::JSGlobalObject& globalObject, const String& markName, std::optional<PerformanceMarkOptions>&& markOptions)
{
    if (!m_userTiming)
        m_userTiming = makeUnique<PerformanceUserTiming>(*this);

    auto mark = m_userTiming->mark(globalObject, markName, WTFMove(markOptions));
    if (mark.hasException())
        return mark.releaseException();

    queueEntry(mark.returnValue().get());
    return mark.releaseReturnValue();
}

void Performance::clearMarks(const String& markName)
{
    if (!m_userTiming)
        m_userTiming = makeUnique<PerformanceUserTiming>(*this);
    m_userTiming->clearMarks(markName);
}

ExceptionOr<Ref<PerformanceMeasure>> Performance::measure(JSC::JSGlobalObject& globalObject, const String& measureName, std::optional<StartOrMeasureOptions>&& startOrMeasureOptions, const String& endMark)
{
    if (!m_userTiming)
        m_userTiming = makeUnique<PerformanceUserTiming>(*this);

    auto measure = m_userTiming->measure(globalObject, measureName, WTFMove(startOrMeasureOptions), endMark);
    if (measure.hasException())
        return measure.releaseException();

    queueEntry(measure.returnValue().get());
    return measure.releaseReturnValue();
}

void Performance::clearMeasures(const String& measureName)
{
    if (!m_userTiming)
        m_userTiming = makeUnique<PerformanceUserTiming>(*this);
    m_userTiming->clearMeasures(measureName);
}

void Performance::removeAllObservers()
{
    for (auto& observer : m_observers)
        observer->disassociate();
    m_observers.clear();
}

void Performance::registerPerformanceObserver(PerformanceObserver& observer)
{
    m_observers.add(&observer);

    if (m_navigationTiming
        && observer.typeFilter().contains(PerformanceEntry::Type::Navigation)
        && !observer.hasNavigationTiming()) {
        observer.queueEntry(*m_navigationTiming);
        observer.addedNavigationTiming();
    }
}

void Performance::unregisterPerformanceObserver(PerformanceObserver& observer)
{
    m_observers.remove(&observer);
}

void Performance::scheduleNavigationObservationTaskIfNeeded()
{
    if (m_navigationTiming)
        scheduleTaskIfNeeded();
}

void Performance::queueEntry(PerformanceEntry& entry)
{
    bool shouldScheduleTask = false;
    for (auto& observer : m_observers) {
        if (observer->typeFilter().contains(entry.performanceEntryType())) {
            observer->queueEntry(entry);
            shouldScheduleTask = true;
        }
    }

    if (!shouldScheduleTask)
        return;

    scheduleTaskIfNeeded();
}

void Performance::scheduleTaskIfNeeded()
{
    if (m_hasScheduledTimingBufferDeliveryTask)
        return;

    auto* context = scriptExecutionContext();
    if (!context)
        return;

    m_hasScheduledTimingBufferDeliveryTask = true;
    context->eventLoop().queueTask(TaskSource::PerformanceTimeline, [protectedThis = Ref { *this }, this] {
        auto* context = scriptExecutionContext();
        if (!context)
            return;

        m_hasScheduledTimingBufferDeliveryTask = false;
        for (auto& observer : copyToVector(m_observers))
            observer->deliver();
    });
}

} // namespace WebCore
