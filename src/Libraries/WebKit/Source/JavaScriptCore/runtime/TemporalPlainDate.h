/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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

class TemporalPlainDate final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalPlainDateSpace<mode>();
    }

    static TemporalPlainDate* create(VM&, Structure*, ISO8601::PlainDate&&);
    static TemporalPlainDate* tryCreateIfValid(JSGlobalObject*, Structure*, ISO8601::PlainDate&&);
    static TemporalPlainDate* tryCreateIfValid(JSGlobalObject*, Structure*, ISO8601::Duration&&);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static ISO8601::PlainDate toPlainDate(JSGlobalObject*, const ISO8601::Duration&);
    static std::array<std::optional<double>, 3> toPartialDate(JSGlobalObject*, JSObject*);

    static TemporalPlainDate* from(JSGlobalObject*, JSValue, std::optional<TemporalOverflow>);

    TemporalCalendar* calendar() { return m_calendar.get(this); }
    ISO8601::PlainDate plainDate() const { return m_plainDate; }

#define JSC_DEFINE_TEMPORAL_PLAIN_DATE_FIELD(name, capitalizedName) \
    decltype(auto) name() const { return m_plainDate.name(); }
    JSC_TEMPORAL_PLAIN_DATE_UNITS(JSC_DEFINE_TEMPORAL_PLAIN_DATE_FIELD);
#undef JSC_DEFINE_TEMPORAL_PLAIN_DATE_FIELD

    ISO8601::PlainDate with(JSGlobalObject*, JSObject* temporalDateLike, JSValue options);

    String monthCode() const;
    uint8_t dayOfWeek() const;
    uint16_t dayOfYear() const;
    uint8_t weekOfYear() const;

    String toString(JSGlobalObject*, JSValue options) const;
    String toString() const
    {
        return ISO8601::temporalDateToString(m_plainDate);
    }

    ISO8601::Duration until(JSGlobalObject*, TemporalPlainDate*, JSValue options);
    ISO8601::Duration since(JSGlobalObject*, TemporalPlainDate*, JSValue options);

    DECLARE_VISIT_CHILDREN;

private:
    TemporalPlainDate(VM&, Structure*, ISO8601::PlainDate&&);
    void finishCreation(VM&);

    template<typename CharacterType>
    static std::optional<ISO8601::PlainDate> parse(StringParsingBuffer<CharacterType>&);
    static ISO8601::PlainDate fromObject(JSGlobalObject*, JSObject*);

    ISO8601::PlainDate m_plainDate;
    LazyProperty<TemporalPlainDate, TemporalCalendar> m_calendar;
};

} // namespace JSC
