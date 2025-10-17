/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

#include "ISO8601.h"
#include "LazyProperty.h"
#include "TemporalCalendar.h"

namespace JSC {

class TemporalPlainDateTime final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalPlainDateTimeSpace<mode>();
    }

    static TemporalPlainDateTime* create(VM&, Structure*, ISO8601::PlainDate&&, ISO8601::PlainTime&&);
    static TemporalPlainDateTime* tryCreateIfValid(JSGlobalObject*, Structure*, ISO8601::PlainDate&&, ISO8601::PlainTime&&);
    static TemporalPlainDateTime* tryCreateIfValid(JSGlobalObject*, Structure*, ISO8601::Duration&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static TemporalPlainDateTime* from(JSGlobalObject*, JSValue, std::optional<TemporalOverflow>);
    static int32_t compare(TemporalPlainDateTime*, TemporalPlainDateTime*);

    TemporalCalendar* calendar() { return m_calendar.get(this); }
    ISO8601::PlainDate plainDate() const { return m_plainDate; }
    ISO8601::PlainTime plainTime() const { return m_plainTime; }

#define JSC_DEFINE_TEMPORAL_PLAIN_DATE_FIELD(name, capitalizedName) \
    decltype(auto) name() const { return m_plainDate.name(); }
    JSC_TEMPORAL_PLAIN_DATE_UNITS(JSC_DEFINE_TEMPORAL_PLAIN_DATE_FIELD);
#undef JSC_DEFINE_TEMPORAL_PLAIN_DATE_FIELD

#define JSC_DEFINE_TEMPORAL_PLAIN_TIME_FIELD(name, capitalizedName) \
    unsigned name() const { return m_plainTime.name(); }
    JSC_TEMPORAL_PLAIN_TIME_UNITS(JSC_DEFINE_TEMPORAL_PLAIN_TIME_FIELD);
#undef JSC_DEFINE_TEMPORAL_PLAIN_TIME_FIELD

    TemporalPlainDateTime* with(JSGlobalObject*, JSObject* temporalDateLike, JSValue options);
    TemporalPlainDateTime* round(JSGlobalObject*, JSValue options);

    String monthCode() const;
    uint8_t dayOfWeek() const;
    uint16_t dayOfYear() const;
    uint8_t weekOfYear() const;

    String toString(JSGlobalObject*, JSValue options) const;
    String toString(std::tuple<Precision, unsigned> precision = { Precision::Auto, 0 }) const
    {
        return ISO8601::temporalDateTimeToString(m_plainDate, m_plainTime, precision);
    }

    DECLARE_VISIT_CHILDREN;

private:
    TemporalPlainDateTime(VM&, Structure*, ISO8601::PlainDate&&, ISO8601::PlainTime&&);
    void finishCreation(VM&);

    ISO8601::PlainDate m_plainDate;
    ISO8601::PlainTime m_plainTime;
    LazyProperty<TemporalPlainDateTime, TemporalCalendar> m_calendar;
};

} // namespace JSC
