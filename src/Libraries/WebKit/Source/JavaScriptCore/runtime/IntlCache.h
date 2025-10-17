/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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

#include <unicode/udatpg.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/CString.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

class IntlCache {
    WTF_MAKE_NONCOPYABLE(IntlCache);
    WTF_MAKE_TZONE_ALLOCATED(IntlCache);
public:
    IntlCache() = default;

    Vector<UChar, 32> getBestDateTimePattern(const CString& locale, std::span<const UChar> skeleton, UErrorCode&);
    Vector<UChar, 32> getFieldDisplayName(const CString& locale, UDateTimePatternField, UDateTimePGDisplayWidth, UErrorCode&);

private:
    UDateTimePatternGenerator* getSharedPatternGenerator(const CString& locale, UErrorCode& status)
    {
        if (m_cachedDateTimePatternGenerator) {
            if (locale == m_cachedDateTimePatternGeneratorLocale)
                return m_cachedDateTimePatternGenerator.get();
        }
        return cacheSharedPatternGenerator(locale, status);
    }

    UDateTimePatternGenerator* cacheSharedPatternGenerator(const CString& locale, UErrorCode&);

    std::unique_ptr<UDateTimePatternGenerator, ICUDeleter<udatpg_close>> m_cachedDateTimePatternGenerator;
    CString m_cachedDateTimePatternGeneratorLocale;
};

} // namespace JSC
