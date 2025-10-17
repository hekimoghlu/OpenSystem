/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include <variant>

namespace JSC {

class TemporalTimeZone final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.temporalTimeZoneSpace<mode>();
    }

    static TemporalTimeZone* createFromID(VM&, Structure*, TimeZoneID);
    static TemporalTimeZone* createFromUTCOffset(VM&, Structure*, int64_t);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    using TimeZone = ISO8601::TimeZone;

    TimeZone timeZone() const { return m_timeZone; }

    static JSObject* from(JSGlobalObject*, JSValue);

private:
    TemporalTimeZone(VM&, Structure*, TimeZone);

    // TimeZoneID or UTC offset.
    TimeZone m_timeZone;
};

} // namespace JSC
