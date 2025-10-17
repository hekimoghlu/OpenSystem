/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#import "MBCFloatingBoardWindow.h"

@implementation MBCFloatingBoardWindow

//
// Adapted from RoundTransparentWindow sample code
//
- (id)initWithContentRect:(NSRect)contentRect styleMask:(NSUInteger)aStyle 
				  backing:(NSBackingStoreType)bufferingType defer:(BOOL)flag 
{
	//
    // Call NSWindow's version of this function, but pass in the all-important
	// value of NSBorderlessWindowMask for the styleMask so that the window 
	// doesn't have a title bar
	//
    MBCFloatingBoardWindow* result = 
		[super initWithContentRect: contentRect 
			   styleMask: NSBorderlessWindowMask 
			   backing: bufferingType defer: flag];
	//
    // Set the background color to clear so that (along with the setOpaque 
	// call below) we can see through the parts of the window that we're not		// drawing into.
	//
    [result setBackgroundColor: [NSColor clearColor]];
	//
    // Let's start with no transparency for all drawing into the window
	//
    [result setAlphaValue:0.999];
	//
    // but let's turn off opaqueness so that we can see through the parts of 
	// the window that we're not drawing into
	//
    [result setOpaque:NO];
	//
    // and while we're at it, make sure the window has a shadow, which will
	// automatically be the shape of our custom content.
	//
    [result setHasShadow: YES];
    return result;
}

//
// Custom windows that use the NSBorderlessWindowMask can't become key by 
// default.  Therefore, controls in such windows won't ever be enabled by 
// default.  Thus, we override this method to change that.
//
- (BOOL) canBecomeKeyWindow
{
    return YES;
}

@end
