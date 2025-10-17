/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class PerformanceEntry;

class PerformanceObserverEntryList : public RefCounted<PerformanceObserverEntryList> {
    WTF_MAKE_TZONE_ALLOCATED(PerformanceObserverEntryList);
public:
    static Ref<PerformanceObserverEntryList> create(Vector<Ref<PerformanceEntry>>&& entries);

    const Vector<Ref<PerformanceEntry>>& getEntries() const { return m_entries; }
    Vector<Ref<PerformanceEntry>> getEntriesByType(const String& entryType) const;
    Vector<Ref<PerformanceEntry>> getEntriesByName(const String& name, const String& entryType) const;

private:
    PerformanceObserverEntryList(Vector<Ref<PerformanceEntry>>&& entries);

    Vector<Ref<PerformanceEntry>> m_entries;
};

} // namespace WebCore
