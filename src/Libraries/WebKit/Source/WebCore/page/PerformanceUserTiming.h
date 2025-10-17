/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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

#include "ExceptionOr.h"
#include "PerformanceMark.h"
#include "PerformanceMeasure.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace JSC {
class JSGlobalObject;
}

namespace WebCore {

class Performance;

using PerformanceEntryMap = UncheckedKeyHashMap<String, Vector<Ref<PerformanceEntry>>>;

class PerformanceUserTiming {
    WTF_MAKE_TZONE_ALLOCATED(PerformanceUserTiming);
public:
    explicit PerformanceUserTiming(Performance&);

    ExceptionOr<Ref<PerformanceMark>> mark(JSC::JSGlobalObject&, const String& markName, std::optional<PerformanceMarkOptions>&&);
    void clearMarks(const String& markName);

    using StartOrMeasureOptions = std::variant<String, PerformanceMeasureOptions>;
    ExceptionOr<Ref<PerformanceMeasure>> measure(JSC::JSGlobalObject&, const String& measureName, std::optional<StartOrMeasureOptions>&&, const String& endMark);
    void clearMeasures(const String& measureName);

    Vector<Ref<PerformanceEntry>> getMarks() const;
    Vector<Ref<PerformanceEntry>> getMeasures() const;

    Vector<Ref<PerformanceEntry>> getMarks(const String& name) const;
    Vector<Ref<PerformanceEntry>> getMeasures(const String& name) const;

    static bool isRestrictedMarkName(const String& markName);

private:
    ExceptionOr<double> convertMarkToTimestamp(const std::variant<String, double>&) const;
    ExceptionOr<double> convertMarkToTimestamp(const String& markName) const;
    ExceptionOr<double> convertMarkToTimestamp(double) const;

    ExceptionOr<Ref<PerformanceMeasure>> measure(const String& measureName, const String& startMark, const String& endMark);
    ExceptionOr<Ref<PerformanceMeasure>> measure(JSC::JSGlobalObject&, const String& measureName, const PerformanceMeasureOptions&);

    WeakRef<Performance, WeakPtrImplWithEventTargetData> m_performance;
    PerformanceEntryMap m_marksMap;
    PerformanceEntryMap m_measuresMap;
};

}
