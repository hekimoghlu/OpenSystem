/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#include "ReportingObserver.h"

#include "DocumentInlines.h"
#include "EventLoop.h"
#include "InspectorInstrumentation.h"
#include "LocalDOMWindow.h"
#include "Report.h"
#include "ReportBody.h"
#include "ReportingObserverCallback.h"
#include "ReportingScope.h"
#include "ScriptExecutionContext.h"
#include "WorkerGlobalScope.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ReportingObserverCallback);
WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ReportingObserver);

static bool isVisibleToReportingObservers(const String& type)
{
    static NeverDestroyed<Vector<String>> visibleTypes(std::initializer_list<String> {
        String { "csp-violation"_s },
        String { "coep"_s },
        String { "deprecation"_s },
        String { "test"_s },
    });
    return visibleTypes->contains(type);
}

Ref<ReportingObserver> ReportingObserver::create(ScriptExecutionContext& scriptExecutionContext, Ref<ReportingObserverCallback>&& callback, Options&& options)
{
    auto reportingObserver = adoptRef(*new ReportingObserver(scriptExecutionContext, WTFMove(callback), WTFMove(options)));
    reportingObserver->suspendIfNeeded();
    return reportingObserver;
}

static WeakPtr<ReportingScope> reportingScopeForContext(ScriptExecutionContext& scriptExecutionContext)
{
    if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext))
        return document->reportingScope();

    if (RefPtr workerGlobalScope = dynamicDowncast<WorkerGlobalScope>(scriptExecutionContext))
        return workerGlobalScope->reportingScope();

    RELEASE_ASSERT_NOT_REACHED();
}

ReportingObserver::ReportingObserver(ScriptExecutionContext& scriptExecutionContext, Ref<ReportingObserverCallback>&& callback, Options&& options)
    : ActiveDOMObject(&scriptExecutionContext)
    , m_reportingScope(reportingScopeForContext(scriptExecutionContext))
    , m_callback(WTFMove(callback))
    , m_types(options.types.value_or(Vector<AtomString>()))
    , m_buffered(options.buffered)
{
}

ReportingObserver::~ReportingObserver() = default;

void ReportingObserver::disconnect()
{
    // https://www.w3.org/TR/reporting-1/#dom-reportingobserver-disconnect
    if (m_reportingScope)
        m_reportingScope->unregisterReportingObserver(*this);
}

void ReportingObserver::observe()
{
    ASSERT(m_reportingScope);
    if (!m_reportingScope)
        return;

    // https://www.w3.org/TR/reporting-1/#dom-reportingobserver-observe
    m_reportingScope->registerReportingObserver(*this);

    if (!m_buffered)
        return;

    m_buffered = false;

    // For each report in globalâ€™s report buffer, queue a task to execute Â§â€¯4.3 Add report to observer with report and the context object.
    m_reportingScope->appendQueuedReportsForRelevantType(*this);
}

auto ReportingObserver::takeRecords() -> Vector<Ref<Report>>
{
    // https://www.w3.org/TR/reporting-1/#dom-reportingobserver-takerecords
    return WTFMove(m_queuedReports);
}

void ReportingObserver::appendQueuedReportIfCorrectType(const Ref<Report>& report)
{
    // https://www.w3.org/TR/reporting-1/#add-report
    // Step 4.3.1
    if (!isVisibleToReportingObservers(report->type()))
        return;
    
    // Step 4.3.2
    if (m_types.size() && !m_types.contains(report->type()))
        return;
    
    // Step 4.3.3:
    m_queuedReports.append(report);

    // Step 4.3.4: (Only enqueue the task once per set of reports)
    if (m_queuedReports.size() > 1)
        return;

    ASSERT(m_reportingScope && scriptExecutionContext() == m_reportingScope->scriptExecutionContext());

    // Step 4.3.4: Queue a task to Â§â€¯4.4
    queueTaskKeepingObjectAlive(*this, TaskSource::Reporting, [protectedThis = Ref { *this }, protectedCallback = Ref { m_callback }] {
        RefPtr context = protectedThis->scriptExecutionContext();
        ASSERT(context);
        if (!context)
            return;

        // Step 4.4: Invoke reporting observers with notify list with a copy of globalâ€™s registered reporting observer list.
        auto reports = protectedThis->takeRecords();

        InspectorInstrumentation::willFireObserverCallback(*context, "ReportingObserver"_s);
        protectedCallback->handleEvent(reports, protectedThis);
        InspectorInstrumentation::didFireObserverCallback(*context);
    });
}

bool ReportingObserver::virtualHasPendingActivity() const
{
    return m_reportingScope && m_reportingScope->containsObserver(*this);
}

ReportingObserverCallback& ReportingObserver::callbackConcurrently()
{
    return m_callback.get();
}

} // namespace WebCore
