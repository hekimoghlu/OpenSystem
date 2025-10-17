/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 23, 2024.
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
#include "IntlObject.h"
#include "JSObject.h"

namespace JSC {

class TemporalCalendar final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalCalendarSpace<mode>();
    }

    static TemporalCalendar* create(VM&, Structure*, CalendarID);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static JSObject* toTemporalCalendarWithISODefault(JSGlobalObject*, JSValue);
    static JSObject* getTemporalCalendarWithISODefault(JSGlobalObject*, JSValue);
    static ISO8601::PlainDate isoDateFromFields(JSGlobalObject*, JSObject*, TemporalOverflow);
    static ISO8601::PlainDate isoDateFromFields(JSGlobalObject*, double year, double month, double day, TemporalOverflow);
    static ISO8601::PlainDate isoDateAdd(JSGlobalObject*, const ISO8601::PlainDate&, const ISO8601::Duration&, TemporalOverflow);
    static ISO8601::Duration isoDateDifference(JSGlobalObject*, const ISO8601::PlainDate&, const ISO8601::PlainDate&, TemporalUnit);
    static int32_t isoDateCompare(const ISO8601::PlainDate&, const ISO8601::PlainDate&);

    CalendarID identifier() const { return m_identifier; }
    bool isISO8601() const { return m_identifier == iso8601CalendarID(); }

    static std::optional<CalendarID> isBuiltinCalendar(StringView);

    static JSObject* from(JSGlobalObject*, JSValue);

    bool equals(JSGlobalObject*, TemporalCalendar*);

private:
    TemporalCalendar(VM&, Structure*, CalendarID);

    CalendarID m_identifier { 0 };
};

} // namespace JSC
