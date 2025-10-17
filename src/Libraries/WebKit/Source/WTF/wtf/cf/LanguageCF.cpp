/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 26, 2024.
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

#include <CoreFoundation/CoreFoundation.h>
#include <mutex>
#include <unicode/uloc.h>
#include <wtf/Assertions.h>
#include <wtf/Logging.h>
#include <wtf/RetainPtr.h>
#include <wtf/spi/cf/CFBundleSPI.h>
#include <wtf/text/TextStream.h>
#include <wtf/text/WTFString.h>

namespace WTF {

#if PLATFORM(MAC)
static void languagePreferencesDidChange(CFNotificationCenterRef, void*, CFStringRef, const void*, CFDictionaryRef)
{
    languageDidChange();
}
#endif

static String httpStyleLanguageCode(CFStringRef language, ShouldMinimizeLanguages shouldMinimizeLanguages)
{
    RetainPtr<CFStringRef> preferredLanguageCode;
    // If we can minimize the language list to reduce fingerprinting, we can afford to be more lossless when canonicalizing the locale list.
    if (shouldMinimizeLanguages == ShouldMinimizeLanguages::No || canMinimizeLanguages())
        preferredLanguageCode = adoptCF(CFLocaleCreateCanonicalLanguageIdentifierFromString(kCFAllocatorDefault, language));
    else {
        SInt32 languageCode;
        SInt32 regionCode;
        SInt32 scriptCode;
        CFStringEncoding stringEncoding;

        // FIXME: This transformation is very wrong:
        // 1. There is no reason why CFBundle localization names would be at all related to language names as used on the Web.
        // 2. Script Manager codes cannot represent all languages that are now supported by the platform, so the conversion is lossy.
        // 3. This should probably match what is sent by the network layer as Accept-Language, but currently, that's implemented separately.
        CFBundleGetLocalizationInfoForLocalization(language, &languageCode, &regionCode, &scriptCode, &stringEncoding);
        preferredLanguageCode = adoptCF(CFBundleCopyLocalizationForLocalizationInfo(languageCode, regionCode, scriptCode, stringEncoding));
    }

    if (!preferredLanguageCode)
        preferredLanguageCode = language;
    auto mutableLanguageCode = adoptCF(CFStringCreateMutableCopy(kCFAllocatorDefault, 0, preferredLanguageCode.get()));

    // Turn a '_' into a '-' if it appears after a 2-letter language code
    if (CFStringGetLength(mutableLanguageCode.get()) >= 3 && CFStringGetCharacterAtIndex(mutableLanguageCode.get(), 2) == '_')
        CFStringReplace(mutableLanguageCode.get(), CFRangeMake(2, 1), CFSTR("-"));

    return mutableLanguageCode.get();
}

void listenForLanguageChangeNotifications()
{
#if PLATFORM(MAC)
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(), nullptr, &languagePreferencesDidChange, CFSTR("AppleLanguagePreferencesChangedNotification"), nullptr, CFNotificationSuspensionBehaviorCoalesce);
    });
#endif
}

Vector<String> platformUserPreferredLanguages(ShouldMinimizeLanguages shouldMinimizeLanguages)
{
    auto platformLanguages = adoptCF(CFLocaleCopyPreferredLanguages());

    LOG_WITH_STREAM(Language, stream << "CFLocaleCopyPreferredLanguages() returned: " << reinterpret_cast<id>(const_cast<CFMutableArrayRef>(platformLanguages.get())));

    if (shouldMinimizeLanguages == ShouldMinimizeLanguages::Yes)
        platformLanguages = minimizedLanguagesFromLanguages(platformLanguages.get());

    LOG_WITH_STREAM(Language, stream << "Minimized languages: " << reinterpret_cast<id>(const_cast<CFMutableArrayRef>(platformLanguages.get())));

    CFIndex platformLanguagesCount = CFArrayGetCount(platformLanguages.get());
    if (!platformLanguagesCount)
        return { "en"_s };

    Vector<String> languages(platformLanguagesCount, [&](size_t i) {
        auto platformLanguage = static_cast<CFStringRef>(CFArrayGetValueAtIndex(platformLanguages.get(), i));
        return httpStyleLanguageCode(platformLanguage, shouldMinimizeLanguages);
    });

    LOG_WITH_STREAM(Language, stream << "After passing through httpStyleLanguageCode: " << languages);

    return languages;
}

} // namespace WTF
