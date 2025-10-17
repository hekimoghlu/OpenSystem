/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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
#import "WKWebEvent.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import <wtf/RetainPtr.h>

@implementation WKWebEvent {
    RetainPtr<UIEvent> _uiEvent;
}

- (instancetype)initWithEvent:(UIEvent *)event
{
    uint16_t keyCode;
    UIKeyboardInputFlags inputFlags;
    NSInteger modifierFlags;
    static auto physicalKeyboardEventClass = NSClassFromString(@"UIPhysicalKeyboardEvent");
    BOOL isHardwareKeyboardEvent = [event isKindOfClass:physicalKeyboardEventClass] && event._hidEvent;
    RetainPtr<UIEvent> uiEvent;
    if (!isHardwareKeyboardEvent) {
        keyCode = 0;
        inputFlags = (UIKeyboardInputFlags)0;
        modifierFlags = 0;
        uiEvent = event;
    } else {
        UIPhysicalKeyboardEvent *keyEvent = (UIPhysicalKeyboardEvent *)event;
        keyCode = keyEvent._keyCode;
        inputFlags = keyEvent._inputFlags;
        modifierFlags = keyEvent._gsModifierFlags;
        uiEvent = adoptNS([keyEvent _cloneEvent]); // UIKit uses a singleton for hardware keyboard events.
    }

    self = [super initWithKeyEventType:([uiEvent _isKeyDown] ? WebEventKeyDown : WebEventKeyUp) timeStamp:[uiEvent timestamp] characters:[uiEvent _modifiedInput] charactersIgnoringModifiers:[uiEvent _unmodifiedInput] modifiers:modifierFlags isRepeating:(inputFlags & kUIKeyboardInputRepeat) withFlags:inputFlags withInputManagerHint:nil keyCode:keyCode isTabKey:[[uiEvent _modifiedInput] isEqualToString:@"\t"]];
    if (!self)
        return nil;

    _uiEvent = WTFMove(uiEvent);

    return self;
}

- (UIEvent *)uiEvent
{
    return _uiEvent.get();
}

@end

#endif // PLATFORM(IOS_FAMILY)
