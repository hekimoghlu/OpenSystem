/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#import "DOMObject.h"

#import "DOMHTMLLinkElementInternal.h"
#import "DOMHTMLStyleElementInternal.h"
#import "DOMInternal.h"
#import "DOMProcessingInstructionInternal.h"
#import "DOMStyleSheetInternal.h"
#import <WebCore/HTMLLinkElement.h>
#import <WebCore/HTMLStyleElement.h>
#import <WebCore/ProcessingInstruction.h>
#import <WebCore/StyleSheet.h>
#import <WebCore/WebScriptObjectPrivate.h>

@implementation DOMObject

// Prevent creation of DOM objects by clients who just "[[xxx alloc] init]".
- (instancetype)init
{
    [NSException raise:NSGenericException format:@"+[%@ init]: should never be used", NSStringFromClass([self class])];

    return nil;
}

- (void)dealloc
{
    if (_internal)
        removeDOMWrapper(_internal);
    [super dealloc];
}

- (id)copyWithZone:(NSZone *)unusedZone
{
    UNUSED_PARAM(unusedZone);
    return [self retain];
}

@end

@implementation DOMObject (DOMLinkStyle)

- (DOMStyleSheet *)sheet
{
    WebCore::StyleSheet* styleSheet;

    if ([self isKindOfClass:[DOMProcessingInstruction class]])
        styleSheet = core(static_cast<DOMProcessingInstruction *>(self))->sheet();
    else if ([self isKindOfClass:[DOMHTMLLinkElement class]])
        styleSheet = core(static_cast<DOMHTMLLinkElement *>(self))->sheet();
    else if ([self isKindOfClass:[DOMHTMLStyleElement class]])
        styleSheet = core(static_cast<DOMHTMLStyleElement *>(self))->sheet();
    else
        return nil;

    return kit(styleSheet);
}

@end
