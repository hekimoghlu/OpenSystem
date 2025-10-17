/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#import "MBCMoveAnimation.h"
#import "MBCPlayer.h"

#include <math.h>

@implementation MBCMoveAnimation

+ (id) moveAnimation:(MBCMove *)move board:(MBCBoard *)board view:(NSView <MBCBoardViewInterface> *)view 
{
    MBCMoveAnimation * a = [[MBCMoveAnimation alloc] init];
    a->fMove	= [move retain];
    a->fPiece	= move->fCommand == kCmdDrop 
		? move->fPiece : [board oldContents:move->fFromSquare];

    [a runWithTime:1.0 view:view];
    
    return a;
}

- (void) startState
{
	[super startState];

	if (fMove->fCommand == kCmdDrop)
		fFrom = [fView squareToPosition:fMove->fPiece+kInHandSquare];
	else
		fFrom	= [fView squareToPosition:fMove->fFromSquare];
    fDelta	= [fView squareToPosition:fMove->fToSquare] - fFrom;

	[fView selectPiece:fPiece at:fMove->fFromSquare to:fMove->fToSquare];
	[fView moveSelectionTo:&fFrom];
}
            
- (void) step: (float)pctDone
{
	MBCPosition	pos = fFrom;
	pos[0] 		   += pctDone*fDelta[0];
	pos[2] 		   += pctDone*fDelta[2];

	[fView moveSelectionTo:&pos];
}

- (void) endState
{
	[fView unselectPiece];
	[[NSNotificationQueue defaultQueue] 
		enqueueNotification:
			[NSNotification 
				notificationWithName:MBCEndMoveNotification
             object:[[[fView window] windowController] document] userInfo:(id)fMove]
		postingStyle: NSPostWhenIdle];
	[super endState];
}

- (void) dealloc
{
	[fMove release];
	[super dealloc];
}

@end

// Local Variables:
// mode:ObjC
// End:
