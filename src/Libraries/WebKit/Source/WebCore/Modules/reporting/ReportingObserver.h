/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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

#include "ActiveDOMObject.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Report;
class ReportingObserverCallback;
class ReportingScope;
class ScriptExecutionContext;

class ReportingObserver final : public RefCounted<ReportingObserver>, public ActiveDOMObject  {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ReportingObserver);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    struct Options {
        std::optional<Vector<AtomString>> types;
        bool buffered { false };
    };

    static Ref<ReportingObserver> create(ScriptExecutionContext&, Ref<ReportingObserverCallback>&&, ReportingObserver::Options&&);

    virtual ~ReportingObserver();

    void observe();
    void disconnect();
    Vector<Ref<Report>> takeRecords();

    void appendQueuedReportIfCorrectType(const Ref<Report>&);

    ReportingObserverCallback& callbackConcurrently();

    bool virtualHasPendingActivity() const final;

private:
    explicit ReportingObserver(ScriptExecutionContext&, Ref<ReportingObserverCallback>&&, ReportingObserver::Options&&);

    WeakPtr<ReportingScope> m_reportingScope;
    Ref<ReportingObserverCallback> m_callback;
    // Instead of storing an Options struct we store the fields separately to save the space overhead of an optional<Vector<AtomString>>
    // which is logically equivalent to an empty vector by the spec.
    const Vector<AtomString> m_types;
    bool m_buffered;
    Vector<Ref<Report>> m_queuedReports;
};

}
