/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#import "MBCMoveTableView.h"

@implementation MBCMoveTableView

- (void)drawGridInClipRect:(NSRect)clipRect
{
	fInGridDrawing	= true;
	[super drawGridInClipRect:clipRect];
	fInGridDrawing	= false;
}

- (NSIndexSet *)columnIndexesInRect:(NSRect)rect
{
	//
	// We only want to draw certain columns
	//
	NSIndexSet * origColumns = [super columnIndexesInRect:rect];
	if (fInGridDrawing) {
		NSMutableIndexSet * filtered = [origColumns mutableCopy];
        if ([[[[self tableColumns] objectAtIndex:0] identifier] isEqual:@"Move"]) {
            //
            // Left to right languages, move # is first
            //
            [filtered removeIndex:2];
            [filtered removeIndex:3];
            [filtered removeIndex:5];
            [filtered removeIndex:6];
        } else {
            //
            // Right to left languages, move # is last
            //
            [filtered removeIndex:1];
            [filtered removeIndex:2];
            [filtered removeIndex:4];
            [filtered removeIndex:5];
        }
		return [filtered autorelease];
	} else {
		return origColumns;
	}
}

@end
