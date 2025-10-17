/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 1, 2021.
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
#include <wtf/unicode/Collator.h>

// FIXME: Merge this with CollatorDefault.cpp into a single Collator.cpp source file.

#if !UCONFIG_NO_COLLATION

#include <mutex>
#include <unicode/ucol.h>
#include <wtf/Lock.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/StringView.h>

#if OS(DARWIN) && USE(CF)
#include <CoreFoundation/CoreFoundation.h>
#include <wtf/RetainPtr.h>
#endif

namespace WTF {

static Lock cachedCollatorLock;
static UCollator* cachedCollator WTF_GUARDED_BY_LOCK(cachedCollatorLock);
static char* cachedCollatorLocale;
static bool cachedCollatorShouldSortLowercaseFirst;

#if !(OS(DARWIN) && USE(CF))

static inline const char* resolveDefaultLocale(const char* locale)
{
    return locale;
}

#else

static inline CString copyShortASCIIString(CFStringRef string)
{
    if (!string)
        return CString(""_s);

    std::span<char> buffer;
    auto result = CString::newUninitialized(CFStringGetLength(string) + 1, buffer);
    if (!CFStringGetCString(string, buffer.data(), buffer.size(), kCFStringEncodingASCII))
        return CString(""_s);
    return result;
}

static CString copyDefaultLocale()
{
#if !PLATFORM(IOS_FAMILY)
    return copyShortASCIIString(static_cast<CFStringRef>(CFLocaleGetValue(adoptCF(CFLocaleCopyCurrent()).get(), kCFLocaleCollatorIdentifier)));
#else
    // FIXME: Documentation claims the code above would work on iOS 4.0 and later. After test that works, we should remove this and use that instead.
    return copyShortASCIIString(adoptCF(static_cast<CFStringRef>(CFPreferencesCopyValue(CFSTR("AppleCollationOrder"), kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost))).get());
#endif
}

static inline const char* resolveDefaultLocale(const char* locale)
{
    if (locale)
        return locale;
    // Since iOS and OS X don't set UNIX locale to match the user's selected locale, the ICU default locale is not the right one.
    // So, instead of passing null to ICU, we pass the name of the user's selected locale.
    static LazyNeverDestroyed<CString> defaultLocale;
    static std::once_flag initializeDefaultLocaleOnce;
    std::call_once(initializeDefaultLocaleOnce, []{
        defaultLocale.construct(copyDefaultLocale());
    });
    return defaultLocale.get().data();
}

#endif

static inline bool localesMatch(const char* a, const char* b)
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    // Two null locales are equal, other locales are compared with strcmp.
    return a == b || (a && b && !strcmp(a, b));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

Collator::Collator(const char* locale, bool shouldSortLowercaseFirst)
{
    UErrorCode status = U_ZERO_ERROR;

    {
        Locker locker { cachedCollatorLock };
        if (cachedCollator && localesMatch(cachedCollatorLocale, locale) && cachedCollatorShouldSortLowercaseFirst == shouldSortLowercaseFirst) {
            m_collator = cachedCollator;
            m_locale = cachedCollatorLocale;
            m_shouldSortLowercaseFirst = shouldSortLowercaseFirst;
            cachedCollator = nullptr;
            cachedCollatorLocale = nullptr;
            return;
        }
    }

    m_collator = ucol_open(resolveDefaultLocale(locale), &status);
    if (U_FAILURE(status)) {
        status = U_ZERO_ERROR;
        m_collator = ucol_open("", &status); // Fall back to Unicode Collation Algorithm.
    }
    ASSERT(U_SUCCESS(status));

    ucol_setAttribute(m_collator, UCOL_CASE_FIRST, shouldSortLowercaseFirst ? UCOL_LOWER_FIRST : UCOL_UPPER_FIRST, &status);
    ASSERT(U_SUCCESS(status));

    ucol_setAttribute(m_collator, UCOL_NORMALIZATION_MODE, UCOL_ON, &status);
    ASSERT(U_SUCCESS(status));

    m_locale = locale ? fastStrDup(locale) : nullptr;
    m_shouldSortLowercaseFirst = shouldSortLowercaseFirst;
}

Collator::~Collator()
{
    Locker locker { cachedCollatorLock };
    if (cachedCollator) {
        ucol_close(cachedCollator);
        fastFree(cachedCollatorLocale);
    }
    cachedCollator = m_collator;
    cachedCollatorLocale = m_locale;
    cachedCollatorShouldSortLowercaseFirst = m_shouldSortLowercaseFirst;
}

static int32_t getIndexLatin1(UCharIterator* iterator, UCharIteratorOrigin origin)
{
    switch (origin) {
    case UITER_START:
        return iterator->start;
    case UITER_CURRENT:
        return iterator->index;
    case UITER_LIMIT:
        return iterator->limit;
    case UITER_ZERO:
        return 0;
    case UITER_LENGTH:
        return iterator->length;
    }
    ASSERT_NOT_REACHED();
    return U_SENTINEL;
}

static int32_t moveLatin1(UCharIterator* iterator, int32_t delta, UCharIteratorOrigin origin)
{
    return iterator->index = getIndexLatin1(iterator, origin) + delta;
}

static UBool hasNextLatin1(UCharIterator* iterator)
{
    return iterator->index < iterator->limit;
}

static UBool hasPreviousLatin1(UCharIterator* iterator)
{
    return iterator->index > iterator->start;
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
static UChar32 currentLatin1(UCharIterator* iterator)
{
    ASSERT(iterator->index >= iterator->start);
    if (iterator->index >= iterator->limit)
        return U_SENTINEL;
    return static_cast<const LChar*>(iterator->context)[iterator->index];
}

static UChar32 nextLatin1(UCharIterator* iterator)
{
    ASSERT(iterator->index >= iterator->start);
    if (iterator->index >= iterator->limit)
        return U_SENTINEL;
    return static_cast<const LChar*>(iterator->context)[iterator->index++];
}

static UChar32 previousLatin1(UCharIterator* iterator)
{
    if (iterator->index <= iterator->start)
        return U_SENTINEL;
    return static_cast<const LChar*>(iterator->context)[--iterator->index];
}
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

static uint32_t getStateLatin1(const UCharIterator* iterator)
{
    return iterator->index;
}

static void setStateLatin1(UCharIterator* iterator, uint32_t state, UErrorCode*)
{
    iterator->index = state;
}

static UCharIterator createLatin1Iterator(std::span<const LChar> characters)
{
    UCharIterator iterator;
    iterator.context = characters.data();
    iterator.length = characters.size();
    iterator.start = 0;
    iterator.index = 0;
    iterator.limit = characters.size();
    iterator.reservedField = 0;
    iterator.getIndex = getIndexLatin1;
    iterator.move = moveLatin1;
    iterator.hasNext = hasNextLatin1;
    iterator.hasPrevious = hasPreviousLatin1;
    iterator.current = currentLatin1;
    iterator.next = nextLatin1;
    iterator.previous = previousLatin1;
    iterator.reservedFn = nullptr;
    iterator.getState = getStateLatin1;
    iterator.setState = setStateLatin1;
    return iterator;
}

UCharIterator createIterator(StringView string)
{
    if (string.is8Bit())
        return createLatin1Iterator(string.span8());
    UCharIterator iterator;
    auto characters = string.span16();
    uiter_setString(&iterator, characters.data(), characters.size());
    return iterator;
}

int Collator::collate(StringView a, StringView b) const
{
    UCharIterator iteratorA = createIterator(a);
    UCharIterator iteratorB = createIterator(b);
    UErrorCode status = U_ZERO_ERROR;
    int result = ucol_strcollIter(m_collator, &iteratorA, &iteratorB, &status);
    ASSERT(U_SUCCESS(status));
    return result;
}

static UCharIterator createIterator(const char8_t* string)
{
    UCharIterator iterator;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    uiter_setUTF8(&iterator, byteCast<char>(string), strlen(byteCast<char>(string)));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    return iterator;
}

int Collator::collate(const char8_t* a, const char8_t* b) const
{
    UCharIterator iteratorA = createIterator(a);
    UCharIterator iteratorB = createIterator(b);
    UErrorCode status = U_ZERO_ERROR;
    int result = ucol_strcollIter(m_collator, &iteratorA, &iteratorB, &status);
    ASSERT(U_SUCCESS(status));
    return result;
}

} // namespace WTF

#endif
