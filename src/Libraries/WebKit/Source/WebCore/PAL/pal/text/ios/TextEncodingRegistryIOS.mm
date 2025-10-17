/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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
#import "TextEncodingRegistry.h"

#if PLATFORM(IOS_FAMILY)

#import <Foundation/Foundation.h>

namespace PAL {

CFStringEncoding webDefaultCFStringEncoding()
{
    // FIXME: we can do better than this hard-coded list once this radar is addressed:
    // <rdar://problem/4433165> Need API that can get preferred web (and mail) encoding(s) w/o region code.
    // Alternatively, we could have our own table of preferred encodings in WebKit, shared with Mac.

    NSArray *preferredLanguages = [NSLocale preferredLanguages];
    if (!preferredLanguages.count)
        return kCFStringEncodingISOLatin1;

    // We pass in the first language as "en" because if preferredLocalizationsFromArray:forPreferences:
    // doesn't find a match, it will return the result of the first value in languagesWithCustomDefaultEncodings.
    // We could really pass anything as this first value, as long as it's not something we're trying to match against
    // below. We don't want to implictly default to "zh-Hans", that's why it's not first in the array.
    NSArray *languagesWithCustomDefaultEncodings = @[ @"en", @"zh-Hans", @"zh-Hant", @"zh-HK", @"ja", @"ko", @"ru" ];
    NSArray *languagesToUse = [NSBundle preferredLocalizationsFromArray:languagesWithCustomDefaultEncodings forPreferences:@[[preferredLanguages firstObject]]];
    if (!languagesToUse.count)
        return kCFStringEncodingISOLatin1;

    NSString *firstLanguage = languagesToUse.firstObject;
    if ([firstLanguage isEqualToString:@"zh-Hans"])
        return kCFStringEncodingEUC_CN;
    if ([firstLanguage isEqualToString:@"zh-Hant"])
        return kCFStringEncodingBig5;
    if ([firstLanguage isEqualToString:@"zh-HK"])
        return kCFStringEncodingBig5_HKSCS_1999;
    if ([firstLanguage isEqualToString:@"ja"])
        return kCFStringEncodingShiftJIS;
    if ([firstLanguage isEqualToString:@"ko"])
        return kCFStringEncodingEUC_KR;
    if ([firstLanguage isEqualToString:@"ru"])
        return kCFStringEncodingWindowsCyrillic;

    return kCFStringEncodingISOLatin1;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
