/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#import "config.h"
#import <wtf/Language.h>

#import <wtf/Logging.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#import <wtf/cocoa/VectorCocoa.h>
#import <wtf/spi/cocoa/NSLocaleSPI.h>
#import <wtf/text/TextStream.h>
#import <wtf/text/WTFString.h>

namespace WTF {

size_t indexOfBestMatchingLanguageInList(const String& language, const Vector<String>& languageList, bool& exactMatch)
{
    auto matchedLanguages = retainPtr([NSLocale matchedLanguagesFromAvailableLanguages:createNSArray(languageList).get() forPreferredLanguages:@[ static_cast<NSString *>(language) ]]);
    if (![matchedLanguages count]) {
        exactMatch = false;
        return notFound;
    }

    String firstMatchedLanguage = [matchedLanguages firstObject];

    exactMatch = language == firstMatchedLanguage;

    auto index = languageList.find(firstMatchedLanguage);
    ASSERT(index < languageList.size());
    return index;
}

LocaleComponents parseLocale(const String& localeIdentifier)
{
    auto locale = retainPtr([NSLocale localeWithLocaleIdentifier:localeIdentifier]);

    return {
        locale.get().languageCode,
        locale.get().scriptCode,
        locale.get().countryCode
    };
}

bool canMinimizeLanguages()
{
    static const bool result = []() -> bool {
        return linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::MinimizesLanguages) && [NSLocale respondsToSelector:@selector(minimizedLanguagesFromLanguages:)];
    }();
    return result;
}

RetainPtr<CFArrayRef> minimizedLanguagesFromLanguages(CFArrayRef languages)
{
    if (!canMinimizeLanguages()) {
        LOG(Language, "Could not minimize languages.");
        return languages;
    }

ALLOW_NEW_API_WITHOUT_GUARDS_BEGIN
    return (__bridge CFArrayRef)[NSLocale minimizedLanguagesFromLanguages:(__bridge NSArray<NSString *> *)languages];
ALLOW_NEW_API_WITHOUT_GUARDS_END
}

void overrideUserPreferredLanguages(const Vector<String>& override)
{
    LOG_WITH_STREAM(Language, stream << "Languages are being overridden to: " << override);
    NSDictionary *existingArguments = [[NSUserDefaults standardUserDefaults] volatileDomainForName:NSArgumentDomain];
    auto newArguments = adoptNS([existingArguments mutableCopy]);
    [newArguments setValue:createNSArray(override).get() forKey:@"AppleLanguages"];
    [[NSUserDefaults standardUserDefaults] setVolatileDomain:newArguments.get() forName:NSArgumentDomain];
    languageDidChange();
}

}
