/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
#import "WebNSAttributedStringExtras.h"

#if PLATFORM(IOS_FAMILY)
#import <UIKit/NSTextAttachment.h>
#else
#import <AppKit/NSTextAttachment.h>
#endif

namespace WebCore {

NSAttributedString *attributedStringByStrippingAttachmentCharacters(NSAttributedString *attributedString)
{
    NSRange attachmentRange;
    NSString *originalString = [attributedString string];
    static NeverDestroyed attachmentCharString = [] {
        unichar chars[2] = { NSAttachmentCharacter, 0 };
        return adoptNS([[NSString alloc] initWithCharacters:chars length:1]);
    }();

    attachmentRange = [originalString rangeOfString:attachmentCharString.get().get()];
    if (attachmentRange.location != NSNotFound && attachmentRange.length > 0) {
        auto newAttributedString = adoptNS([attributedString mutableCopy]);

        while (attachmentRange.location != NSNotFound && attachmentRange.length > 0) {
            [newAttributedString replaceCharactersInRange:attachmentRange withString:@""];
            attachmentRange = [[newAttributedString string] rangeOfString:attachmentCharString.get().get()];
        }
        return newAttributedString.autorelease();
    }

    return attributedString;
}

}
