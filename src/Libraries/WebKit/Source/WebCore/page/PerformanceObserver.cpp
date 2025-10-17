/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#include "PerformanceObserver.h"

#include "Document.h"
#include "InspectorInstrumentation.h"
#include "LocalDOMWindow.h"
#include "Performance.h"
#include "PerformanceObserverEntryList.h"
#include "WorkerGlobalScope.h"

namespace WebCore {

PerformanceObserver::PerformanceObserver(ScriptExecutionContext& scriptExecutionContext, Ref<PerformanceObserverCallback>&& callback)
    : m_callback(WTFMove(callback))
{
    if (RefPtr document = dynamicDowncast<Document>(scriptExecutionContext)) {
        if (auto* window = document->domWindow())
            m_performance = &window->performance();
    } else if (RefPtr workerGlobalScope = dynamicDowncast<WorkerGlobalScope>(scriptExecutionContext))
        m_performance = &workerGlobalScope->performance();
    else
        ASSERT_NOT_REACHED();
}

RefPtr<Performance> PerformanceObserver::protectedPerformance() const
{
    return m_performance;
}

void PerformanceObserver::disassociate()
{
    m_performance = nullptr;
    m_registered = false;
}

ExceptionOr<void> PerformanceObserver::observe(Init&& init)
{
    if (!m_performance)
        return Exception { ExceptionCode::TypeError };

    bool isBuffered = false;
    OptionSet<PerformanceEntry::Type> filter;
    if (init.entryTypes) {
        if (init.type)
            return Exception { ExceptionCode::TypeError, "either entryTypes or type must be provided"_s };
        if (m_registered && m_isTypeObserver)
            return Exception { ExceptionCode::InvalidModificationError, "observer type can't be changed once registered"_s };
        for (auto& entryType : *init.entryTypes) {
            if (auto type = PerformanceEntry::parseEntryTypeString(entryType))
                filter.add(*type);
        }
        if (filter.isEmpty())
            return { };
        m_typeFilter = filter;
    } else {
        if (!init.type)
            return Exception { ExceptionCode::TypeError, "no type or entryTypes were provided"_s };
        if (m_registered && !m_isTypeObserver)
            return Exception { ExceptionCode::InvalidModificationError, "observer type can't be changed once registered"_s };
        m_isTypeObserver = true;
        if (auto type = PerformanceEntry::parseEntryTypeString(*init.type))
            filter.add(*type);
        else
            return { };
        if (init.buffered) {
            isBuffered = true;
            auto oldSize = m_entriesToDeliver.size();
            protectedPerformance()->appendBufferedEntriesByType(*init.type, m_entriesToDeliver, *this);
            auto entriesToDeliver = m_entriesToDeliver.mutableSpan();
            auto begin = entriesToDeliver.begin();
            auto oldEnd = entriesToDeliver.subspan(oldSize).begin();
            auto end = entriesToDeliver.end();
            std::stable_sort(oldEnd, end, PerformanceEntry::startTimeCompareLessThan);
            std::inplace_merge(begin, oldEnd, end, PerformanceEntry::startTimeCompareLessThan);
        }
        m_typeFilter.add(filter);
    }

    if (!m_registered) {
        protectedPerformance()->registerPerformanceObserver(*this);
        m_registered = true;
    }
    if (isBuffered)
        deliver();

    return { };
}

Vector<Ref<PerformanceEntry>> PerformanceObserver::takeRecords()
{
    return std::exchange(m_entriesToDeliver, { });
}

void PerformanceObserver::disconnect()
{
    if (RefPtr performance = m_performance)
        performance->unregisterPerformanceObserver(*this);

    m_registered = false;
    m_entriesToDeliver.clear();
    m_typeFilter = { };
}

void PerformanceObserver::queueEntry(PerformanceEntry& entry)
{
    m_entriesToDeliver.append(entry);
}

void PerformanceObserver::deliver()
{
    if (m_entriesToDeliver.isEmpty())
        return;

    auto* context = m_callback->scriptExecutionContext();
    if (!context)
        return;

    Vector<Ref<PerformanceEntry>> entries = std::exchange(m_entriesToDeliver, { });
    auto list = PerformanceObserverEntryList::create(WTFMove(entries));

    InspectorInstrumentation::willFireObserverCallback(*context, "PerformanceObserver"_s);
    m_callback->handleEvent(*this, list, *this);
    InspectorInstrumentation::didFireObserverCallback(*context);
}

Vector<String> PerformanceObserver::supportedEntryTypes(ScriptExecutionContext& context)
{
    Vector<String> entryTypes = {
        "mark"_s,
        "measure"_s,
        "navigation"_s,
    };

    if (RefPtr document = dynamicDowncast<Document>(context); document && document->supportsPaintTiming())
        entryTypes.append("paint"_s);

    entryTypes.append("resource"_s);

    return entryTypes;
}

} // namespace WebCore
