/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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

#ifdef __OBJC__
#import <Foundation/Foundation.h>
typedef NSString *WebLocalizedStringType;
#else
#ifdef __cplusplus
class NSBundle;
#else
typedef struct NSBundle NSBundle;
#endif
typedef CFStringRef WebLocalizedStringType;
#endif

typedef struct {
    const char *identifier;
    __unsafe_unretained NSBundle *bundle;
} WebLocalizableStringsBundle;

#ifdef __cplusplus
extern "C" {
#endif

WebLocalizedStringType WebLocalizedString(WebLocalizableStringsBundle* bundle, const char* key);

#ifdef __cplusplus
}
#endif

static inline __attribute__((format_arg(3))) WebLocalizedStringType WebLocalizedStringWithValue(WebLocalizableStringsBundle* bundle, const char* key, const char* value)
{
    return WebLocalizedString(bundle, key);
}

#ifdef FRAMEWORK_NAME

#define LOCALIZABLE_STRINGS_BUNDLE(F) LOCALIZABLE_STRINGS_BUNDLE_HELPER(F)
#define LOCALIZABLE_STRINGS_BUNDLE_HELPER(F) F ## LocalizableStringsBundle

__attribute__((visibility("hidden")))
extern WebLocalizableStringsBundle LOCALIZABLE_STRINGS_BUNDLE(FRAMEWORK_NAME);

#define UI_STRING(string, comment) WebLocalizedStringWithValue(&LOCALIZABLE_STRINGS_BUNDLE(FRAMEWORK_NAME), string, string)
#define UI_STRING_KEY(string, key, comment) WebLocalizedStringWithValue(&LOCALIZABLE_STRINGS_BUNDLE(FRAMEWORK_NAME), key, string)

#else

#define UI_STRING(string, comment) WebLocalizedStringWithValue(0, string, string)
#define UI_STRING_KEY(string, key, comment) WebLocalizedStringWithValue(0, key, string)

#endif
