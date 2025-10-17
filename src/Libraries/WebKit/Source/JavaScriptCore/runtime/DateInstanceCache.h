/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

#include "PureNaN.h"
#include <array>
#include <wtf/GregorianDateTime.h>
#include <wtf/HashFunctions.h>
#include <wtf/RefCounted.h>

namespace JSC {

class DateInstanceData : public RefCounted<DateInstanceData> {
public:
    static Ref<DateInstanceData> create() { return adoptRef(*new DateInstanceData); }

    static constexpr ptrdiff_t offsetOfGregorianDateTimeCachedForMS() { return OBJECT_OFFSETOF(DateInstanceData, m_gregorianDateTimeCachedForMS); }
    static constexpr ptrdiff_t offsetOfCachedGregorianDateTime() { return OBJECT_OFFSETOF(DateInstanceData, m_cachedGregorianDateTime); }
    static constexpr ptrdiff_t offsetOfGregorianDateTimeUTCCachedForMS() { return OBJECT_OFFSETOF(DateInstanceData, m_gregorianDateTimeUTCCachedForMS); }
    static constexpr ptrdiff_t offsetOfCachedGregorianDateTimeUTC() { return OBJECT_OFFSETOF(DateInstanceData, m_cachedGregorianDateTimeUTC); }

    double m_gregorianDateTimeCachedForMS { PNaN };
    GregorianDateTime m_cachedGregorianDateTime;
    double m_gregorianDateTimeUTCCachedForMS { PNaN };
    GregorianDateTime m_cachedGregorianDateTimeUTC;

private:
    DateInstanceData() = default;
};

class DateInstanceCache {
public:
    DateInstanceCache()
    {
        reset();
    }

    void reset()
    {
        for (size_t i = 0; i < cacheSize; ++i)
            m_cache[i].key = PNaN;
    }

    DateInstanceData* add(double d)
    {
        CacheEntry& entry = lookup(d);
        if (d == entry.key)
            return entry.value.get();

        entry.key = d;
        entry.value = DateInstanceData::create();
        return entry.value.get();
    }

private:
    static const size_t cacheSize = 16;

    struct CacheEntry {
        double key;
        RefPtr<DateInstanceData> value;
    };

    CacheEntry& lookup(double d) { return m_cache[WTF::FloatHash<double>::hash(d) & (cacheSize - 1)]; }

    std::array<CacheEntry, cacheSize> m_cache;
};

} // namespace JSC
