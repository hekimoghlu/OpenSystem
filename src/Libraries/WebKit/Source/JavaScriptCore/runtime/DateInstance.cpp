/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "config.h"
#include "DateInstance.h"

#include "JSCInlines.h"
#include "JSDateMath.h"

namespace JSC {

const ClassInfo DateInstance::s_info = { "Date"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(DateInstance) };

DateInstance::DateInstance(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

void DateInstance::finishCreation(VM& vm, double time)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    m_internalNumber = timeClip(time);
}

const GregorianDateTime* DateInstance::calculateGregorianDateTime(DateCache& cache) const
{
    double milli = internalNumber();
    if (std::isnan(milli))
        return nullptr;

    if (!m_data)
        m_data = cache.cachedDateInstanceData(milli);

    if (m_data->m_gregorianDateTimeCachedForMS != milli) {
        cache.msToGregorianDateTime(milli, WTF::LocalTime, m_data->m_cachedGregorianDateTime);
        m_data->m_gregorianDateTimeCachedForMS = milli;
    }
    return &m_data->m_cachedGregorianDateTime;
}

const GregorianDateTime* DateInstance::calculateGregorianDateTimeUTC(DateCache& cache) const
{
    double milli = internalNumber();
    if (std::isnan(milli))
        return nullptr;

    if (!m_data)
        m_data = cache.cachedDateInstanceData(milli);

    if (m_data->m_gregorianDateTimeUTCCachedForMS != milli) {
        cache.msToGregorianDateTime(milli, WTF::UTCTime, m_data->m_cachedGregorianDateTimeUTC);
        m_data->m_gregorianDateTimeUTCCachedForMS = milli;
    }
    return &m_data->m_cachedGregorianDateTimeUTC;
}

} // namespace JSC
