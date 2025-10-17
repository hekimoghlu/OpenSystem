/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
#import "PopupMenu.h"

#if PLATFORM(MAC)

#import "NSMenuSPI.h"
#import <wtf/RetainPtr.h>

@interface NSMenu (WebPrivate)
- (id)_menuImpl;
@end

@interface NSObject (WebPrivate)
- (void)popUpMenu:(NSMenu*)menu atLocation:(NSPoint)location width:(CGFloat)width forView:(NSView*)view withSelectedItem:(NSInteger)selectedItem withFont:(NSFont*)font withFlags:(NSPopUpMenuFlags)flags withOptions:(NSDictionary *)options;
@end

namespace PAL {

void popUpMenu(NSMenu *menu, NSPoint location, float width, NSView *view, int selectedItem, NSFont *font, NSControlSize controlSize, bool usesCustomAppearance)
{
    NSRect adjustedPopupBounds = [view.window convertRectToScreen:[view convertRect:view.bounds toView:nil]];
    if (controlSize != NSControlSizeMini) {
        adjustedPopupBounds.origin.x -= 3;
        adjustedPopupBounds.origin.y -= 1;
        adjustedPopupBounds.size.width += 6;
    }

    // These numbers were extracted from visual inspection as the menu animates shut.
    NSSize labelOffset = NSMakeSize(11, 1);
    if (menu.userInterfaceLayoutDirection == NSUserInterfaceLayoutDirectionRightToLeft)
        labelOffset = NSMakeSize(24, 1);

    auto options = adoptNS([@{
        NSPopUpMenuPopupButtonBounds : [NSValue valueWithRect:adjustedPopupBounds],
        NSPopUpMenuPopupButtonLabelOffset : [NSValue valueWithSize:labelOffset],
        NSPopUpMenuPopupButtonSize : @(controlSize)
    } mutableCopy]);

    if (usesCustomAppearance)
        [options setObject:@"" forKey:NSPopUpMenuPopupButtonWidget];

    [[menu _menuImpl] popUpMenu:menu atLocation:location width:width forView:view withSelectedItem:selectedItem withFont:font withFlags:(usesCustomAppearance ? 0 : NSPopUpMenuIsPopupButton) withOptions:options.get()];
}

} // namespace PAL

#endif // PLATFORM(MAC)
