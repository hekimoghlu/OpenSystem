/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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

class TemporalPlainTime final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalPlainTimeSpace<mode>();
    }

    static TemporalPlainTime* create(VM&, Structure*, ISO8601::PlainTime&&);
    static TemporalPlainTime* tryCreateIfValid(JSGlobalObject*, Structure*, ISO8601::Duration&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static ISO8601::PlainTime toPlainTime(JSGlobalObject*, const ISO8601::Duration&);
    static ISO8601::Duration roundTime(ISO8601::PlainTime, double increment, TemporalUnit, RoundingMode, std::optional<double> dayLengthNs);
    static ISO8601::Duration toTemporalTimeRecord(JSGlobalObject*, JSObject*, bool skipRelevantPropertyCheck = false);
    static std::array<std::optional<double>, numberOfTemporalPlainTimeUnits> toPartialTime(JSGlobalObject*, JSObject*, bool skipRelevantPropertyCheck = false);
    static ISO8601::PlainTime regulateTime(JSGlobalObject*, ISO8601::Duration&&, TemporalOverflow);
    static ISO8601::Duration addTime(const ISO8601::PlainTime&, const ISO8601::Duration&);

    static TemporalPlainTime* from(JSGlobalObject*, JSValue, std::optional<TemporalOverflow>);
    static int32_t compare(const ISO8601::PlainTime&, const ISO8601::PlainTime&);

    TemporalCalendar* calendar() { return m_calendar.get(this); }
    ISO8601::PlainTime plainTime() const { return m_plainTime; }

#define JSC_DEFINE_TEMPORAL_PLAIN_TIME_FIELD(name, capitalizedName) \
    unsigned name() const { return m_plainTime.name(); }
    JSC_TEMPORAL_PLAIN_TIME_UNITS(JSC_DEFINE_TEMPORAL_PLAIN_TIME_FIELD);
#undef JSC_DEFINE_TEMPORAL_PLAIN_TIME_FIELD

    ISO8601::PlainTime with(JSGlobalObject*, JSObject* temporalTimeLike, JSValue options) const;
    ISO8601::PlainTime round(JSGlobalObject*, JSValue options) const;
    String toString(JSGlobalObject*, JSValue options) const;
    String toString(std::tuple<Precision, unsigned> precision = { Precision::Auto, 0 }) const
    {
        return ISO8601::temporalTimeToString(m_plainTime, precision);
    }

    ISO8601::Duration until(JSGlobalObject*, TemporalPlainTime*, JSValue options) const;
    ISO8601::Duration since(JSGlobalObject*, TemporalPlainTime*, JSValue options) const;

    DECLARE_VISIT_CHILDREN;

private:
    TemporalPlainTime(VM&, Structure*, ISO8601::PlainTime&&);
    void finishCreation(VM&);

    template<typename CharacterType>
    static std::optional<ISO8601::PlainTime> parse(StringParsingBuffer<CharacterType>&);
    static ISO8601::PlainTime fromObject(JSGlobalObject*, JSObject*);

    ISO8601::PlainTime m_plainTime;
    LazyProperty<TemporalPlainTime, TemporalCalendar> m_calendar;
};

} // namespace JSC
