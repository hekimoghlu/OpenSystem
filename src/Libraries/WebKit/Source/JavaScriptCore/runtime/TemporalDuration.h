/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 18, 2025.
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

namespace JSC {

class TemporalDuration final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalDurationSpace<mode>();
    }

    static TemporalDuration* create(VM&, Structure*, ISO8601::Duration&&);
    static TemporalDuration* tryCreateIfValid(JSGlobalObject*, ISO8601::Duration&&, Structure* = nullptr);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static TemporalDuration* toTemporalDuration(JSGlobalObject*, JSValue);
    static ISO8601::Duration toLimitedDuration(JSGlobalObject*, JSValue, std::initializer_list<TemporalUnit> disallowedUnits);
    static TemporalDuration* from(JSGlobalObject*, JSValue);
    static JSValue compare(JSGlobalObject*, JSValue, JSValue);

#define JSC_DEFINE_TEMPORAL_DURATION_FIELD(name, capitalizedName) \
    double name##s() const { return m_duration.name##s(); } \
    void set##capitalizedName##s(double value) { m_duration.set##capitalizedName##s(value); }
    JSC_TEMPORAL_UNITS(JSC_DEFINE_TEMPORAL_DURATION_FIELD);
#undef JSC_DEFINE_TEMPORAL_DURATION_FIELD

    int sign() const { return sign(m_duration); }

    ISO8601::Duration with(JSGlobalObject*, JSObject* durationLike) const;
    ISO8601::Duration negated() const;
    ISO8601::Duration abs() const;
    ISO8601::Duration add(JSGlobalObject*, JSValue) const;
    ISO8601::Duration subtract(JSGlobalObject*, JSValue) const;
    ISO8601::Duration round(JSGlobalObject*, JSValue options) const;
    double total(JSGlobalObject*, JSValue options) const;
    String toString(JSGlobalObject*, JSValue options) const;
    String toString(JSGlobalObject* globalObject, std::tuple<Precision, unsigned> precision = { Precision::Auto, 0 }) const { return toString(globalObject, m_duration, precision); }

    static ISO8601::Duration fromDurationLike(JSGlobalObject*, JSObject*);
    static ISO8601::Duration toISO8601Duration(JSGlobalObject*, JSValue);

    static int sign(const ISO8601::Duration&);
    static double round(ISO8601::Duration&, double increment, TemporalUnit, RoundingMode);
    static std::optional<double> balance(ISO8601::Duration&, TemporalUnit largestUnit);

private:
    TemporalDuration(VM&, Structure*, ISO8601::Duration&&);
    DECLARE_DEFAULT_FINISH_CREATION;

    template<typename CharacterType>
    static std::optional<ISO8601::Duration> parse(StringParsingBuffer<CharacterType>&);

    static String toString(JSGlobalObject*, const ISO8601::Duration&, std::tuple<Precision, unsigned> precision);

    ISO8601::Duration m_duration;
};

} // namespace JSC
