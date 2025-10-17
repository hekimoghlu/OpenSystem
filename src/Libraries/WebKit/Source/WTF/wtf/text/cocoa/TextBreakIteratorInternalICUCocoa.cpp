/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include <wtf/text/TextBreakIteratorInternalICU.h>

#include <array>
#include <wtf/RetainPtr.h>
#include <wtf/cf/TypeCastsCF.h>
#include <wtf/text/TextBreakIterator.h>

namespace WTF {

// Buffer sized to hold ASCII locale ID strings up to 32 characters long.
using LocaleIDBuffer = std::array<char, 33>;

TextBreakIterator::Backing TextBreakIterator::mapModeToBackingIterator(StringView string, std::span<const UChar> priorContext, Mode mode, ContentAnalysis contentAnalysis, const AtomString& locale)
{
    return switchOn(mode, [string, priorContext, contentAnalysis, &locale](TextBreakIterator::LineMode lineMode) -> TextBreakIterator::Backing {
        if (contentAnalysis == ContentAnalysis::Linguistic && lineMode.behavior == LineMode::Behavior::Default)
            return TextBreakIteratorCF(string, priorContext, TextBreakIteratorCF::Mode::LineBreak, locale);
        return TextBreakIteratorICU(string, priorContext, TextBreakIteratorICU::LineMode { lineMode.behavior }, locale);
    }, [string, priorContext, &locale](TextBreakIterator::CaretMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorCF(string, priorContext, TextBreakIteratorCF::Mode::ComposedCharacter, locale);
    }, [string, priorContext, &locale](TextBreakIterator::DeleteMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorCF(string, priorContext, TextBreakIteratorCF::Mode::BackwardDeletion, locale);
    }, [string, priorContext, &locale](TextBreakIterator::CharacterMode) -> TextBreakIterator::Backing {
        return TextBreakIteratorCF(string, priorContext, TextBreakIteratorCF::Mode::ComposedCharacter, locale);
    });
}

TextBreakIterator::TextBreakIterator(StringView string, std::span<const UChar> priorContext, Mode mode, ContentAnalysis contentAnalysis, const AtomString& locale)
    : m_backing(mapModeToBackingIterator(string, priorContext, mode, contentAnalysis, locale))
    , m_mode(mode)
    , m_locale(locale)
{
}

static RetainPtr<CFStringRef> textBreakLocalePreference()
{
    return dynamic_cf_cast<CFStringRef>(adoptCF(CFPreferencesCopyValue(CFSTR("AppleTextBreakLocale"),
        kCFPreferencesAnyApplication, kCFPreferencesCurrentUser, kCFPreferencesAnyHost)));
}

static RetainPtr<CFStringRef> topLanguagePreference()
{
    auto languagesArray = adoptCF(CFLocaleCopyPreferredLanguages());
    if (!languagesArray || !CFArrayGetCount(languagesArray.get()))
        return nullptr;
    return static_cast<CFStringRef>(CFArrayGetValueAtIndex(languagesArray.get(), 0));
}

static LocaleIDBuffer localeIDInBuffer(CFStringRef string)
{
    // Empty string means "root locale", and is what we use if we can't get a preference.
    LocaleIDBuffer buffer;
    if (!string || !CFStringGetCString(string, buffer.data(), buffer.size(), kCFStringEncodingASCII))
        buffer.front() = '\0';
    return buffer;
}

const char* currentSearchLocaleID()
{
    static const auto buffer = localeIDInBuffer(topLanguagePreference().get());
    return buffer.data();
}

static RetainPtr<CFStringRef> textBreakLocale()
{
    // If there is no text break locale, use the top language preference.
    auto locale = textBreakLocalePreference();
    if (!locale)
        return topLanguagePreference();
    if (auto canonicalLocale = adoptCF(CFLocaleCreateCanonicalLanguageIdentifierFromString(kCFAllocatorDefault, locale.get())))
        return canonicalLocale;
    return locale;
}

const char* currentTextBreakLocaleID()
{
    static const auto buffer = localeIDInBuffer(textBreakLocale().get());
    return buffer.data();
}

}
