/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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
#import "WKSyntheticFlagsChangedWebEvent.h"

#if PLATFORM(IOS_FAMILY)

#import <pal/spi/cocoa/IOKitSPI.h>
#import <pal/spi/ios/GraphicsServicesSPI.h>

@implementation WKSyntheticFlagsChangedWebEvent

- (instancetype)initWithKeyCode:(uint16_t)keyCode modifiers:(WebEventFlags)modifiers keyDown:(BOOL)keyDown
{
    self = [super initWithKeyEventType:(keyDown ? WebEventKeyDown : WebEventKeyUp) timeStamp:GSCurrentEventTimestamp() characters:@"" charactersIgnoringModifiers:@"" modifiers:modifiers isRepeating:NO withFlags:WebEventKeyboardInputModifierFlagsChanged withInputManagerHint:nil keyCode:keyCode isTabKey:(keyCode == kHIDUsage_KeyboardTab)];
    return self;
}

- (instancetype)initWithCapsLockState:(BOOL)keyDown
{
    return [self initWithKeyCode:kHIDUsage_KeyboardCapsLock modifiers:(keyDown ? WebEventFlagMaskLeftCapsLockKey : 0) keyDown:keyDown];
}

- (instancetype)initWithShiftState:(BOOL)keyDown
{
    return [self initWithKeyCode:kHIDUsage_KeyboardLeftShift modifiers:(keyDown ? WebEventFlagMaskLeftShiftKey : 0) keyDown:keyDown];
}

@end

#endif // PLATFORM(IOS_FAMILY)
