/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

namespace JSC {

class DateInstance final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell* cell)
    {
        static_cast<DateInstance*>(cell)->DateInstance::~DateInstance();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.dateInstanceSpace();
    }

    static DateInstance* create(VM& vm, Structure* structure, double date)
    {
        DateInstance* instance = new (NotNull, allocateCell<DateInstance>(vm)) DateInstance(vm, structure);
        instance->finishCreation(vm, date);
        return instance;
    }

    static DateInstance* create(VM& vm, Structure* structure)
    {
        DateInstance* instance = new (NotNull, allocateCell<DateInstance>(vm)) DateInstance(vm, structure);
        instance->finishCreation(vm);
        return instance;
    }

    double internalNumber() const { return m_internalNumber; }
    void setInternalNumber(double value) { m_internalNumber = value; }

    DECLARE_EXPORT_INFO;

    const GregorianDateTime* gregorianDateTime(DateCache& cache) const
    {
        if (m_data && m_data->m_gregorianDateTimeCachedForMS == internalNumber())
            return &m_data->m_cachedGregorianDateTime;
        return calculateGregorianDateTime(cache);
    }

    const GregorianDateTime* gregorianDateTimeUTC(DateCache& cache) const
    {
        if (m_data && m_data->m_gregorianDateTimeUTCCachedForMS == internalNumber())
            return &m_data->m_cachedGregorianDateTimeUTC;
        return calculateGregorianDateTimeUTC(cache);
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static constexpr ptrdiff_t offsetOfInternalNumber() { return OBJECT_OFFSETOF(DateInstance, m_internalNumber); }
    static constexpr ptrdiff_t offsetOfData() { return OBJECT_OFFSETOF(DateInstance, m_data); }

private:
    JS_EXPORT_PRIVATE DateInstance(VM&, Structure*);

    DECLARE_DEFAULT_FINISH_CREATION;
    JS_EXPORT_PRIVATE void finishCreation(VM&, double);
    JS_EXPORT_PRIVATE const GregorianDateTime* calculateGregorianDateTime(DateCache&) const;
    JS_EXPORT_PRIVATE const GregorianDateTime* calculateGregorianDateTimeUTC(DateCache&) const;

    double m_internalNumber { PNaN };
    mutable RefPtr<DateInstanceData> m_data;
};

} // namespace JSC
