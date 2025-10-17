/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 27, 2023.
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
#import "MBCAnimation.h"
#import "MBCBoardWin.h"

#include <algorithm>

using std::min;

@implementation MBCAnimation

- (void) scheduleNextStep
{
    // This method sets up a timer to perform doStep: message on the current threadâ€™s run loop.
    // The message is dequed if the run loop is running and in one of the specified modes; otherwise,
    // the timer waits until the run loop is in one of those modes.
    [self performSelector:@selector(doStep:) withObject:nil afterDelay:0.010 inModes:@[NSRunLoopCommonModes]];
}

- (void) startState 		
{
	[fView startAnimation];
}

- (void) step: (float)pctDone	{}

- (void) endState			
{
	[fView animationDone];
}

- (void) doStep:(id)arg
{
	struct timeval now;

	gettimeofday(&now, NULL);
	float	elapsedTime			= 
		now.tv_sec - fStart.tv_sec 
		+ 0.000001f * (now.tv_usec - fStart.tv_usec);
	float	elapsed				= min(elapsedTime/fTime, 1.0f);
	//
	// Prevent excessive jerks on slow hardware
	//
	if (elapsed-fLastElapsed > 0.5f)
		elapsed = 1.0f;
	[self step:elapsed];
	fLastElapsed = elapsed;
	if (elapsed >= 1.0f) {
		[self endState];
        [[[fView window] windowController] endAnimation];
		[self release];
		[fView setNeedsDisplay:YES];
	} else {
		[self scheduleNextStep];
		[fView drawNow];
	}
}

- (void) runWithTime:(float)seconds view:(NSView<MBCBoardViewInterface> *)view
{
	gettimeofday(&fStart, NULL);
	fTime			= seconds;
	fView			= view;
	fLastElapsed	= 0.0f;
    [self startState];
	[self doStep:nil];
}

- (void)cancel
{
    [NSObject cancelPreviousPerformRequestsWithTarget:self];
    fStart.tv_sec = 0;
    [self doStep:self];
}

@end

// Local Variables:
// mode:ObjC
// End:
