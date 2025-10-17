/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#include "JSObject.h"
#include "TemporalDuration.h"
#include "TemporalObject.h"
#include "VM.h"
#include <wtf/Packed.h>

namespace JSC {

class TemporalInstant final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalInstantSpace<mode>();
    }

    static TemporalInstant* create(VM&, Structure*, ISO8601::ExactTime);
    static TemporalInstant* tryCreateIfValid(JSGlobalObject*, ISO8601::ExactTime, Structure* = nullptr);
    static TemporalInstant* tryCreateIfValid(JSGlobalObject*, JSValue, Structure* = nullptr);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    static TemporalInstant* toInstant(JSGlobalObject*, JSValue);
    static TemporalInstant* from(JSGlobalObject*, JSValue);
    static TemporalInstant* fromEpochMilliseconds(JSGlobalObject*, JSValue);
    static TemporalInstant* fromEpochNanoseconds(JSGlobalObject*, JSValue);
    static JSValue compare(JSGlobalObject*, JSValue, JSValue);

    ISO8601::ExactTime exactTime() const { return m_exactTime.get(); }

    ISO8601::Duration difference(JSGlobalObject*, TemporalInstant*, JSValue options) const;
    ISO8601::ExactTime round(JSGlobalObject*, JSValue options) const;
    String toString(JSGlobalObject*, JSValue options) const;
    String toString(JSObject* timeZone = nullptr, PrecisionData precision = { { Precision::Auto, 0 }, TemporalUnit::Nanosecond, 1 }) const
    {
        return toString(exactTime(), timeZone, precision);
    }

private:
    TemporalInstant(VM&, Structure*, ISO8601::ExactTime);

    template<typename CharacterType>
    static std::optional<ISO8601::ExactTime> parse(StringParsingBuffer<CharacterType>&);
    static ISO8601::ExactTime fromObject(JSGlobalObject*, JSObject*);

    static String toString(ISO8601::ExactTime, JSObject* timeZone, PrecisionData);

    Packed<ISO8601::ExactTime> m_exactTime;
};

} // namespace JSC
