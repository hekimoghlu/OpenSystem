/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#if !PLATFORM(IOS_FAMILY)

#import <WebKitLegacy/WebNSEventExtras.h>

@implementation NSEvent (WebExtras)

-(BOOL)_web_isKeyEvent:(unichar)key
{
    int type = [self type];
    if (type != NSEventTypeKeyDown && type != NSEventTypeKeyUp)
        return NO;
    
    NSString *chars = [self charactersIgnoringModifiers];
    if ([chars length] != 1)
        return NO;
    
    unichar c = [chars characterAtIndex:0];
    if (c != key)
        return NO;
    
    return YES;
}

- (BOOL)_web_isDeleteKeyEvent
{
    const unichar deleteKey = NSDeleteCharacter;
    const unichar deleteForwardKey = NSDeleteFunctionKey;
    return [self _web_isKeyEvent:deleteKey] || [self _web_isKeyEvent:deleteForwardKey];
}

- (BOOL)_web_isEscapeKeyEvent
{
    const unichar escapeKey = 0x001b;
    return [self _web_isKeyEvent:escapeKey];
}

- (BOOL)_web_isOptionTabKeyEvent
{
    return ([self modifierFlags] & NSEventModifierFlagOption) && [self _web_isTabKeyEvent];
}

- (BOOL)_web_isReturnOrEnterKeyEvent
{
    const unichar enterKey = NSEnterCharacter;
    const unichar returnKey = NSCarriageReturnCharacter;
    return [self _web_isKeyEvent:enterKey] || [self _web_isKeyEvent:returnKey];
}

- (BOOL)_web_isTabKeyEvent
{
    const unichar tabKey = 0x0009;
    const unichar shiftTabKey = 0x0019;
    return [self _web_isKeyEvent:tabKey] || [self _web_isKeyEvent:shiftTabKey];
}

@end

#endif // !PLATFORM(IOS_FAMILY)
