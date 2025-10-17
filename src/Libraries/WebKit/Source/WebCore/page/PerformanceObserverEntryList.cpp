/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#include "PerformanceObserverEntryList.h"

#include "PerformanceEntry.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PerformanceObserverEntryList);

Ref<PerformanceObserverEntryList> PerformanceObserverEntryList::create(Vector<Ref<PerformanceEntry>>&& entries)
{
    return adoptRef(*new PerformanceObserverEntryList(WTFMove(entries)));
}

PerformanceObserverEntryList::PerformanceObserverEntryList(Vector<Ref<PerformanceEntry>>&& entries)
    : m_entries(WTFMove(entries))
{
    ASSERT(!m_entries.isEmpty());

    std::stable_sort(m_entries.begin(), m_entries.end(), PerformanceEntry::startTimeCompareLessThan);
}

Vector<Ref<PerformanceEntry>> PerformanceObserverEntryList::getEntriesByType(const String& entryType) const
{
    return getEntriesByName(String(), entryType);
}

Vector<Ref<PerformanceEntry>> PerformanceObserverEntryList::getEntriesByName(const String& name, const String& entryType) const
{
    Vector<Ref<PerformanceEntry>> entries;

    // PerformanceObservers can only be registered for valid types.
    // So if the incoming entryType is an unknown type, there will be no matches.
    std::optional<PerformanceEntry::Type> type;
    if (!entryType.isNull()) {
        type = PerformanceEntry::parseEntryTypeString(entryType);
        if (!type)
            return entries;
    }

    for (auto& entry : m_entries) {
        if (!name.isNull() && entry->name() != name)
            continue;
        if (type && entry->performanceEntryType() != *type)
            continue;
        entries.append(entry.copyRef());
    }

    return entries;
}

} // namespace WebCore
