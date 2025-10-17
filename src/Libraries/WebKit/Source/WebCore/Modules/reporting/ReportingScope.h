/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#include "ViolationReportType.h"
#include <wtf/Deque.h>
#include <wtf/HashCountedSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WTF {
class String;
}

namespace WebCore {

class Report;
class ReportingObserver;
class ScriptExecutionContext;

class WEBCORE_EXPORT ReportingScope final : public RefCountedAndCanMakeWeakPtr<ReportingScope>, public ContextDestructionObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(ReportingScope, WEBCORE_EXPORT);
public:
    static Ref<ReportingScope> create(ScriptExecutionContext&);
    virtual ~ReportingScope();

    void removeAllObservers();
    void clearReports();

    void registerReportingObserver(ReportingObserver&);
    void unregisterReportingObserver(ReportingObserver&);
    void notifyReportObservers(Ref<Report>&&);
    void appendQueuedReportsForRelevantType(ReportingObserver&);

    bool containsObserver(const ReportingObserver&) const;

    static MemoryCompactRobinHoodHashMap<String, String> parseReportingEndpointsFromHeader(const String&, const URL& baseURL);
    void parseReportingEndpoints(const String&, const URL& baseURL);

    String endpointURIForToken(const String&) const;

    void generateTestReport(String&& message, String&& group);

private:
    explicit ReportingScope(ScriptExecutionContext&);

    Vector<Ref<ReportingObserver>> m_reportingObservers;
    Deque<Ref<Report>> m_queuedReports;
    HashCountedSet<ViolationReportType, IntHash<ViolationReportType>, WTF::StrongEnumHashTraits<ViolationReportType>> m_queuedReportTypeCounts;

    MemoryCompactRobinHoodHashMap<String, String> m_reportingEndpoints;
};

}
