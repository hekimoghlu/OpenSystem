/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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

#import "WebNSControlExtras.h"

@implementation NSControl (WebExtras)

- (void)sizeToFitAndAdjustWindowHeight
{
    NSRect frame = [self frame];

    NSSize bestSize = [[self cell] cellSizeForBounds:NSMakeRect(0.0f, 0.0f, frame.size.width, 10000.0f)];
    
    float heightDelta = bestSize.height - frame.size.height;

    frame.size.height += heightDelta;
    frame.origin.y    -= heightDelta;
    [self setFrame:frame];

    NSWindow *window = [self window];
    NSRect windowFrame = [window frame];

    windowFrame.size.height += heightDelta;
    [window setFrame:windowFrame display:NO];
}

@end

#endif
