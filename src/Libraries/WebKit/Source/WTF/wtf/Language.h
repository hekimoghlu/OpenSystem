/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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

#include <wtf/Forward.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#import <CoreFoundation/CoreFoundation.h>
#import <wtf/RetainPtr.h>
#endif

namespace WTF {

enum class ShouldMinimizeLanguages : bool { No, Yes };

struct LocaleComponents {
    String languageCode;
    String scriptCode;
    String regionCode;
};

WTF_EXPORT_PRIVATE String defaultLanguage(ShouldMinimizeLanguages = ShouldMinimizeLanguages::Yes); // Thread-safe.
WTF_EXPORT_PRIVATE Vector<String> userPreferredLanguages(ShouldMinimizeLanguages = ShouldMinimizeLanguages::Yes); // Thread-safe, returns BCP 47 language tags.
WTF_EXPORT_PRIVATE void overrideUserPreferredLanguages(const Vector<String>&);
WTF_EXPORT_PRIVATE size_t indexOfBestMatchingLanguageInList(const String& language, const Vector<String>& languageList, bool& exactMatch);
WTF_EXPORT_PRIVATE bool userPrefersSimplifiedChinese();
WTF_EXPORT_PRIVATE LocaleComponents parseLocale(const String&);

// Called from platform specific code when the user's preferred language(s) change.
WTF_EXPORT_PRIVATE void languageDidChange();

// The observer function will be called when system language changes.
typedef void (*LanguageChangeObserverFunction)(void* context);
WTF_EXPORT_PRIVATE void addLanguageChangeObserver(void* context, LanguageChangeObserverFunction);
WTF_EXPORT_PRIVATE void removeLanguageChangeObserver(void* context);
WTF_EXPORT_PRIVATE String displayNameForLanguageLocale(const String&);

Vector<String> platformUserPreferredLanguages(ShouldMinimizeLanguages = ShouldMinimizeLanguages::Yes);

#if PLATFORM(COCOA)
bool canMinimizeLanguages();
WTF_EXPORT_PRIVATE void listenForLanguageChangeNotifications();
RetainPtr<CFArrayRef> minimizedLanguagesFromLanguages(CFArrayRef);
#endif

} // namespace WTF

using WTF::ShouldMinimizeLanguages;
using WTF::defaultLanguage;
using WTF::userPreferredLanguages;
using WTF::overrideUserPreferredLanguages;
using WTF::indexOfBestMatchingLanguageInList;
using WTF::userPrefersSimplifiedChinese;
using WTF::parseLocale;
using WTF::addLanguageChangeObserver;
using WTF::removeLanguageChangeObserver;
using WTF::displayNameForLanguageLocale;

