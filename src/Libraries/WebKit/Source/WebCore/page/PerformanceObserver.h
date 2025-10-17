/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#include "PerformanceEntry.h"
#include "PerformanceObserverCallback.h"
#include <wtf/OptionSet.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Performance;
class ScriptExecutionContext;

class PerformanceObserver : public RefCounted<PerformanceObserver> {
public:
    struct Init {
        std::optional<Vector<String>> entryTypes;
        std::optional<String> type;
        bool buffered;
    };

    static Ref<PerformanceObserver> create(ScriptExecutionContext& context, Ref<PerformanceObserverCallback>&& callback)
    {
        return adoptRef(*new PerformanceObserver(context, WTFMove(callback)));
    }

    static Vector<String> supportedEntryTypes(ScriptExecutionContext&);

    void disassociate();

    ExceptionOr<void> observe(Init&&);
    void disconnect();
    Vector<Ref<PerformanceEntry>> takeRecords();

    OptionSet<PerformanceEntry::Type> typeFilter() const { return m_typeFilter; }

    bool hasNavigationTiming() const { return m_hasNavigationTiming; }
    void addedNavigationTiming() { m_hasNavigationTiming = true; }

    void queueEntry(PerformanceEntry&);
    void deliver();

    bool isRegistered() const { return m_registered; }
    PerformanceObserverCallback& callback() { return m_callback.get(); }

private:
    PerformanceObserver(ScriptExecutionContext&, Ref<PerformanceObserverCallback>&&);

    RefPtr<Performance> protectedPerformance() const;

    RefPtr<Performance> m_performance;
    Vector<Ref<PerformanceEntry>> m_entriesToDeliver;
    Ref<PerformanceObserverCallback> m_callback;
    OptionSet<PerformanceEntry::Type> m_typeFilter;
    bool m_registered { false };
    bool m_isTypeObserver { false };
    bool m_hasNavigationTiming { false };
};

} // namespace WebCore
