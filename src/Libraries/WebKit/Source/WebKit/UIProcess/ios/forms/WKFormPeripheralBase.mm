/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#import "WKFormPeripheralBase.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKContentViewInteraction.h"
#import <pal/spi/cocoa/IOKitSPI.h>

@implementation WKFormPeripheralBase {
    RetainPtr<NSObject <WKFormControl>> _control;
}

- (instancetype)initWithView:(WKContentView *)view control:(RetainPtr<NSObject <WKFormControl>>&&)control
{
    if (!(self = [super init]))
        return nil;

    _view = view;
    _control = WTFMove(control);
    return self;
}

- (void)beginEditing
{
    if (_editing)
        return;

    _editing = YES;
    [_control controlBeginEditing];
}

- (void)updateEditing
{
    if (!_editing)
        return;

    [_control controlUpdateEditing];
}

- (void)endEditing
{
    if (!_editing)
        return;

    _editing = NO;
    [_control controlEndEditing];
}

- (UIView *)assistantView
{
    return [_control controlView];
}

- (NSObject <WKFormControl> *)control
{
    return _control.get();
}

- (BOOL)handleKeyEvent:(UIEvent *)event
{
    ASSERT(event._hidEvent);
    if ([_control respondsToSelector:@selector(controlHandleKeyEvent:)]) {
        if ([_control controlHandleKeyEvent:event])
            return YES;
    }
    if (!event._isKeyDown)
        return NO;
    UIPhysicalKeyboardEvent *keyEvent = (UIPhysicalKeyboardEvent *)event;
    if (keyEvent._inputFlags & kUIKeyboardInputModifierFlagsChanged)
        return NO;
    if (_editing && (keyEvent._keyCode == kHIDUsage_KeyboardEscape || [keyEvent._unmodifiedInput isEqualToString:UIKeyInputEscape])) {
        [_view accessoryDone];
        return YES;
    }
    if (!_editing && keyEvent._keyCode == kHIDUsage_KeyboardSpacebar) {
        [_view accessoryOpen];
        return YES;
    }
    return NO;
}

@end

#endif // PLATFORM(IOS_FAMILY)
