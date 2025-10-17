/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#include "IntlCache.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IntlCache);

UDateTimePatternGenerator* IntlCache::cacheSharedPatternGenerator(const CString& locale, UErrorCode& status)
{
    auto generator = std::unique_ptr<UDateTimePatternGenerator, ICUDeleter<udatpg_close>>(udatpg_open(locale.data(), &status));
    if (U_FAILURE(status))
        return nullptr;
    m_cachedDateTimePatternGeneratorLocale = locale;
    m_cachedDateTimePatternGenerator = WTFMove(generator);
    return m_cachedDateTimePatternGenerator.get();
}

Vector<UChar, 32> IntlCache::getBestDateTimePattern(const CString& locale, std::span<const UChar> skeleton, UErrorCode& status)
{
    // Always use ICU date format generator, rather than our own pattern list and matcher.
    auto sharedGenerator = getSharedPatternGenerator(locale, status);
    if (U_FAILURE(status))
        return { };
    Vector<UChar, 32> patternBuffer;
    status = callBufferProducingFunction(udatpg_getBestPatternWithOptions, sharedGenerator, skeleton.data(), skeleton.size(), UDATPG_MATCH_HOUR_FIELD_LENGTH, patternBuffer);
    if (U_FAILURE(status))
        return { };
    return patternBuffer;
}

Vector<UChar, 32> IntlCache::getFieldDisplayName(const CString& locale, UDateTimePatternField field, UDateTimePGDisplayWidth width, UErrorCode& status)
{
    auto sharedGenerator = getSharedPatternGenerator(locale, status);
    if (U_FAILURE(status))
        return { };
    Vector<UChar, 32> buffer;
    status = callBufferProducingFunction(udatpg_getFieldDisplayName, sharedGenerator, field, width, buffer);
    if (U_FAILURE(status))
        return { };
    return buffer;
}

} // namespace JSC
