/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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
#import "TextAlternativeWithRange.h"

#if USE(APPKIT)
#import <AppKit/NSTextAlternatives.h>
#elif PLATFORM(IOS_FAMILY)
#import <pal/spi/cocoa/NSAttributedStringSPI.h>
#import <pal/spi/ios/UIKitSPI.h>
#endif

namespace WebCore {

TextAlternativeWithRange::TextAlternativeWithRange(PlatformTextAlternatives *textAlternatives, NSRange aRange)
    : range(aRange)
    , alternatives(textAlternatives)
{
}

#if PLATFORM(MAC)

void collectDictationTextAlternatives(NSAttributedString *string, Vector<TextAlternativeWithRange>& alternatives) {
    NSRange effectiveRange = NSMakeRange(0, 0);
    NSUInteger length = [string length];
    do {
        NSDictionary *attributes = [string attributesAtIndex:effectiveRange.location effectiveRange:&effectiveRange];
        if (!attributes)
            break;
        NSTextAlternatives *textAlternatives = [attributes objectForKey:NSTextAlternativesAttributeName];
        if (textAlternatives)
            alternatives.append(TextAlternativeWithRange(textAlternatives, effectiveRange));
        effectiveRange.location = NSMaxRange(effectiveRange);
    } while (effectiveRange.location < length);
}

#endif // PLATFORM(MAC)

} // namespace WebCore
