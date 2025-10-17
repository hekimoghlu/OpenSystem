/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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
#include <wtf/Language.h>

#include <mutex>
#include <windows.h>
#include <wtf/Lock.h>
#include <wtf/Vector.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>
#include <wtf/text/win/WCharStringExtras.h>

namespace WTF {

static Lock platformLanguageMutex;

static String localeInfo(LCTYPE localeType, const String& fallback)
{
    LANGID langID = GetUserDefaultUILanguage();
    int localeChars = GetLocaleInfo(langID, localeType, nullptr, 0);
    if (!localeChars)
        return fallback;
    std::span<UChar> localeNameBuf;
    String localeName = String::createUninitialized(localeChars, localeNameBuf);
    localeChars = GetLocaleInfo(langID, localeType, wcharFrom(localeNameBuf.data()), localeChars);
    if (!localeChars)
        return fallback;
    if (localeName.isEmpty())
        return fallback;

    return localeName.left(localeName.length() - 1);
}

static String platformLanguage()
{
    Locker locker { platformLanguageMutex };

    static String computedDefaultLanguage;
    if (!computedDefaultLanguage.isEmpty())
        return computedDefaultLanguage.isolatedCopy();

    String languageName = localeInfo(LOCALE_SISO639LANGNAME, "en"_s);
    String countryName = localeInfo(LOCALE_SISO3166CTRYNAME, String());

    if (countryName.isEmpty())
        computedDefaultLanguage = languageName;
    else
        computedDefaultLanguage = makeString(languageName, '-', countryName);

    return computedDefaultLanguage;
}

Vector<String> platformUserPreferredLanguages(ShouldMinimizeLanguages)
{
    return { platformLanguage() };
}

} // namespace WTF
