/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#import "MBCBoardMTLView.h"

struct MBCPosition;

@interface MBCBoardMTLView (Mouse)

/*!
 @abstract approximateBoundsOfSquare:
 @param square Square code on the board
 @discussion Will convert the given board square from world coordinates to screen space coordinates as a NSRect
 */
- (NSRect)approximateBoundsOfSquare:(MBCSquare)square;

/*!
 @abstract mouseToPosition:
 @param mouse Screen position for the mouse
 @discussion Will convert the given mouse screen position to coordinates in the world
 */
- (MBCPosition)mouseToPosition:(NSPoint)mouse;

/*!
 @abstract mouseDown:
 @param event The event for mouseDown
 @discussion Called when initially click mouse
 */
- (void)mouseDown:(NSEvent *)event;

/*!
 @abstract mouseMoved:
 @param event The event for mouseMoved
 @discussion Called when move the mouse
 */
- (void)mouseMoved:(NSEvent *)event;

/*!
 @abstract mouseUp:
 @param event The event for mouseUp
 @discussion Called when release mouse click
 */
- (void)mouseUp:(NSEvent *)event;

/*!
 @abstract dragAndRedraw:forceRedraw:
 @param force
 @discussion Called to handle mouse drag while clicking.  Depending upon which region is
 clicked will either do piece selection/moving or rotate the board (via camera movement).
 */
- (void)dragAndRedraw:(NSEvent *)event forceRedraw:(BOOL)force;

@end
