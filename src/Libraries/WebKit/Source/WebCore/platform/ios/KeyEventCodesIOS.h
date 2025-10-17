/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 25, 2024.
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

#if TARGET_OS_IPHONE

#import <WebCore/WAKAppKitStubs.h>

#if WAK_APPKIT_API_AVAILABLE_MACCATALYST
#import <AppKit/NSEvent.h>
#else

// Unicodes we reserve for function keys on the keyboard,
// OpenStep reserves the range 0xF700-0xF8FF for this purpose.
// The availability of various keys will be system dependent.

enum {
    NSUpArrowFunctionKey      = 0xF700,
    NSDownArrowFunctionKey    = 0xF701,
    NSLeftArrowFunctionKey    = 0xF702,
    NSRightArrowFunctionKey   = 0xF703,
    NSF1FunctionKey           = 0xF704,
    NSF2FunctionKey           = 0xF705,
    NSF3FunctionKey           = 0xF706,
    NSF4FunctionKey           = 0xF707,
    NSF5FunctionKey           = 0xF708,
    NSF6FunctionKey           = 0xF709,
    NSF7FunctionKey           = 0xF70A,
    NSF8FunctionKey           = 0xF70B,
    NSF9FunctionKey           = 0xF70C,
    NSF10FunctionKey          = 0xF70D,
    NSF11FunctionKey          = 0xF70E,
    NSF12FunctionKey          = 0xF70F,
    NSF13FunctionKey          = 0xF710,
    NSF14FunctionKey          = 0xF711,
    NSF15FunctionKey          = 0xF712,
    NSF16FunctionKey          = 0xF713,
    NSF17FunctionKey          = 0xF714,
    NSF18FunctionKey          = 0xF715,
    NSF19FunctionKey          = 0xF716,
    NSF20FunctionKey          = 0xF717,
    NSF21FunctionKey          = 0xF718,
    NSF22FunctionKey          = 0xF719,
    NSF23FunctionKey          = 0xF71A,
    NSF24FunctionKey          = 0xF71B,
    NSF25FunctionKey          = 0xF71C,
    NSF26FunctionKey          = 0xF71D,
    NSF27FunctionKey          = 0xF71E,
    NSF28FunctionKey          = 0xF71F,
    NSF29FunctionKey          = 0xF720,
    NSF30FunctionKey          = 0xF721,
    NSF31FunctionKey          = 0xF722,
    NSF32FunctionKey          = 0xF723,
    NSF33FunctionKey          = 0xF724,
    NSF34FunctionKey          = 0xF725,
    NSF35FunctionKey          = 0xF726,
    NSInsertFunctionKey       = 0xF727,
    NSDeleteFunctionKey       = 0xF728,
    NSHomeFunctionKey         = 0xF729,
    NSBeginFunctionKey        = 0xF72A,
    NSEndFunctionKey          = 0xF72B,
    NSPageUpFunctionKey       = 0xF72C,
    NSPageDownFunctionKey     = 0xF72D,
    NSPrintScreenFunctionKey  = 0xF72E,
    NSScrollLockFunctionKey   = 0xF72F,
    NSPauseFunctionKey        = 0xF730,
    NSSysReqFunctionKey       = 0xF731,
    NSBreakFunctionKey        = 0xF732,
    NSResetFunctionKey        = 0xF733,
    NSStopFunctionKey         = 0xF734,
    NSMenuFunctionKey         = 0xF735,
    NSUserFunctionKey         = 0xF736,
    NSSystemFunctionKey       = 0xF737,
    NSPrintFunctionKey        = 0xF738,
    NSClearLineFunctionKey    = 0xF739,
    NSClearDisplayFunctionKey = 0xF73A,
    NSInsertLineFunctionKey   = 0xF73B,
    NSDeleteLineFunctionKey   = 0xF73C,
    NSInsertCharFunctionKey   = 0xF73D,
    NSDeleteCharFunctionKey   = 0xF73E,
    NSPrevFunctionKey         = 0xF73F,
    NSNextFunctionKey         = 0xF740,
    NSSelectFunctionKey       = 0xF741,
    NSExecuteFunctionKey      = 0xF742,
    NSUndoFunctionKey         = 0xF743,
    NSRedoFunctionKey         = 0xF744,
    NSFindFunctionKey         = 0xF745,
    NSHelpFunctionKey         = 0xF746,
    NSModeSwitchFunctionKey   = 0xF747
};
#endif // WAK_APPKIT_API_AVAILABLE_MACCATALYST

#if WAK_APPKIT_API_AVAILABLE_MACCATALYST
#import <AppKit/NSText.h>
#else
enum {
    NSParagraphSeparatorCharacter = 0x2029,
    NSLineSeparatorCharacter = 0x2028,
    NSTabCharacter = 0x0009,
    NSFormFeedCharacter = 0x000c,
    NSNewlineCharacter = 0x000a,
    NSCarriageReturnCharacter = 0x000d,
    NSEnterCharacter = 0x0003,
    NSBackspaceCharacter = 0x0008,
    NSBackTabCharacter = 0x0019,
    NSDeleteCharacter = 0x007f
};
#endif // WAK_APPKIT_API_AVAILABLE_MACCATALYST

#endif // TARGET_OS_IPHONE
