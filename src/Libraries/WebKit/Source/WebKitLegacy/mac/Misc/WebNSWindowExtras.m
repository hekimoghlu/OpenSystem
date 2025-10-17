/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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

#import "WebNSWindowExtras.h"

@implementation NSWindow (WebExtras)

- (void)centerOverMainWindow
{
    NSRect frameToCenterOver;
    if ([NSApp mainWindow]) {
        frameToCenterOver = [[NSApp mainWindow] frame];
    } else {
        frameToCenterOver = [[NSScreen mainScreen] visibleFrame];
    }
    
    NSSize size = [self frame].size;
    NSPoint origin;
    origin.y = NSMaxY(frameToCenterOver)
        - (frameToCenterOver.size.height - size.height) / 3
        - size.height;
    origin.x = frameToCenterOver.origin.x
        + (frameToCenterOver.size.width - size.width) / 2;
    [self setFrameOrigin:origin];
}

- (void)makeResponder:(NSResponder *)responder firstResponderIfDescendantOfView:(NSView *)view
{
    if ([responder isKindOfClass:[NSView class]] && [(id)responder isDescendantOf:view])
        [self makeFirstResponder:responder];
}

@end

#endif
