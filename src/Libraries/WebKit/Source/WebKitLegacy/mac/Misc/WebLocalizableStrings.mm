/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 17, 2025.
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
#import <WebKitLegacy/WebLocalizableStrings.h>

#import <wtf/Assertions.h>
#import <wtf/MainThread.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>

WebLocalizableStringsBundle WebKitLocalizableStringsBundle = { "com.apple.WebKit", 0 };

NSString *WebLocalizedString(WebLocalizableStringsBundle *stringsBundle, const char *key)
{
    NSBundle *bundle;
    if (stringsBundle == NULL) {
        static LazyNeverDestroyed<RetainPtr<NSBundle>> mainBundle;
        static std::once_flag flag;
        std::call_once(flag, [] () {
            mainBundle.construct([NSBundle mainBundle]);
        });
        ASSERT(mainBundle.get());
        bundle = mainBundle.get().get();
    } else {
        bundle = stringsBundle->bundle;
        if (bundle == nil) {
            bundle = [NSBundle bundleWithIdentifier:[NSString stringWithUTF8String:stringsBundle->identifier]];
            ASSERT(bundle);
            stringsBundle->bundle = bundle;
        }
    }
    NSString *notFound = @"localized string not found";
    auto keyString = adoptCF(CFStringCreateWithCStringNoCopy(NULL, key, kCFStringEncodingUTF8, kCFAllocatorNull));
    NSString *result = [bundle localizedStringForKey:(__bridge NSString *)keyString.get() value:notFound table:nil];
    ASSERT_WITH_MESSAGE(result != notFound, "could not find localizable string %s in bundle", key);
    return result;
}
