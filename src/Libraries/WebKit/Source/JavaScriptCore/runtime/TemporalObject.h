/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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

#include "JSObject.h"
#include <variant>
#include <wtf/Int128.h>

namespace JSC {

#define JSC_TEMPORAL_PLAIN_DATE_UNITS(macro) \
    macro(year, Year) \
    macro(month, Month) \
    macro(day, Day) \


#define JSC_TEMPORAL_PLAIN_TIME_UNITS(macro) \
    macro(hour, Hour) \
    macro(minute, Minute) \
    macro(second, Second) \
    macro(millisecond, Millisecond) \
    macro(microsecond, Microsecond) \
    macro(nanosecond, Nanosecond) \


#define JSC_TEMPORAL_UNITS(macro) \
    macro(year, Year) \
    macro(month, Month) \
    macro(week, Week) \
    macro(day, Day) \
    JSC_TEMPORAL_PLAIN_TIME_UNITS(macro) \


enum class TemporalUnit : uint8_t {
#define JSC_DEFINE_TEMPORAL_UNIT_ENUM(name, capitalizedName) capitalizedName,
    JSC_TEMPORAL_UNITS(JSC_DEFINE_TEMPORAL_UNIT_ENUM)
#undef JSC_DEFINE_TEMPORAL_UNIT_ENUM
};
#define JSC_COUNT_TEMPORAL_UNITS(name, capitalizedName) + 1
static constexpr unsigned numberOfTemporalUnits = 0 JSC_TEMPORAL_UNITS(JSC_COUNT_TEMPORAL_UNITS);
static constexpr unsigned numberOfTemporalPlainDateUnits = 0 JSC_TEMPORAL_PLAIN_DATE_UNITS(JSC_COUNT_TEMPORAL_UNITS);
static constexpr unsigned numberOfTemporalPlainTimeUnits = 0 JSC_TEMPORAL_PLAIN_TIME_UNITS(JSC_COUNT_TEMPORAL_UNITS);
#undef JSC_COUNT_TEMPORAL_UNITS

extern const TemporalUnit temporalUnitsInTableOrder[numberOfTemporalUnits];

class TemporalObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(TemporalObject, Base);
        return &vm.plainObjectSpace();
    }

    static TemporalObject* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*);

    DECLARE_INFO;

private:
    TemporalObject(VM&, Structure*);
    void finishCreation(VM&);
};

enum class RoundingMode : uint8_t {
    Ceil,
    Floor,
    Expand,
    Trunc,
    HalfCeil,
    HalfFloor,
    HalfExpand,
    HalfTrunc,
    HalfEven
};

enum class Precision : uint8_t {
    Minute,
    Fixed,
    Auto,
};

struct PrecisionData {
    std::tuple<Precision, unsigned> precision;
    TemporalUnit unit;
    unsigned increment;
};

enum class UnitGroup : uint8_t {
    DateTime,
    Date,
    Time,
};

double nonNegativeModulo(double x, double y);
WTF::String ellipsizeAt(unsigned maxLength, const WTF::String&);
PropertyName temporalUnitPluralPropertyName(VM&, TemporalUnit);
PropertyName temporalUnitSingularPropertyName(VM&, TemporalUnit);
std::optional<TemporalUnit> temporalUnitType(StringView);
std::optional<TemporalUnit> temporalLargestUnit(JSGlobalObject*, JSObject* options, std::initializer_list<TemporalUnit> disallowedUnits, TemporalUnit autoValue);
std::optional<TemporalUnit> temporalSmallestUnit(JSGlobalObject*, JSObject* options, std::initializer_list<TemporalUnit> disallowedUnits);
std::tuple<TemporalUnit, TemporalUnit, RoundingMode, double> extractDifferenceOptions(JSGlobalObject*, JSValue, UnitGroup, TemporalUnit defaultSmallestUnit, TemporalUnit defaultLargestUnit);
std::optional<unsigned> temporalFractionalSecondDigits(JSGlobalObject*, JSObject* options);
PrecisionData secondsStringPrecision(JSGlobalObject*, JSObject* options);
RoundingMode temporalRoundingMode(JSGlobalObject*, JSObject*, RoundingMode);
RoundingMode negateTemporalRoundingMode(RoundingMode);
void formatSecondsStringFraction(StringBuilder&, unsigned fraction, std::tuple<Precision, unsigned>);
void formatSecondsStringPart(StringBuilder&, unsigned second, unsigned fraction, PrecisionData);
std::optional<double> maximumRoundingIncrement(TemporalUnit);
double temporalRoundingIncrement(JSGlobalObject*, JSObject* options, std::optional<double> dividend, bool inclusive);
double roundNumberToIncrement(double, double increment, RoundingMode);
Int128 roundNumberToIncrement(Int128, Int128 increment, RoundingMode);
void rejectObjectWithCalendarOrTimeZone(JSGlobalObject*, JSObject*);

enum class TemporalOverflow : bool {
    Constrain,
    Reject,
};

TemporalOverflow toTemporalOverflow(JSGlobalObject*, JSObject*);

} // namespace JSC
