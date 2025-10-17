/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
/*
 */

#import "MBCRemotePlayer.h"
#import "MBCDocument.h"

@implementation MBCRemotePlayer 

- (void) removeChessObservers
{
    if (!fHasObservers)
        return;
    
    NSNotificationCenter * notificationCenter = [NSNotificationCenter defaultCenter];
    [notificationCenter removeObserver:self name:MBCWhiteMoveNotification object:nil];
    [notificationCenter removeObserver:self name:MBCBlackMoveNotification object:nil];
    [notificationCenter removeObserver:self name:MBCGameEndNotification object:nil];
    [notificationCenter removeObserver:self name:MBCTakebackNotification object:nil];
    
    fHasObservers = NO;
}

- (void)dealloc
{
    [self removeChessObservers];
    [super dealloc];
}

- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay
{
    [self removeChessObservers];
    NSNotificationCenter * notificationCenter = [NSNotificationCenter defaultCenter];
	switch (sideToPlay) {
    default:
    case kWhiteSide:
        [notificationCenter
         addObserver:self
            selector:@selector(opponentMoved:)
                name:MBCBlackMoveNotification
              object:fDocument];
        break;
    case kBlackSide:
        [notificationCenter 
         addObserver:self
            selector:@selector(opponentMoved:)
                name:MBCWhiteMoveNotification
              object:fDocument];
        break;
	}
	[notificationCenter 
     addObserver:self
     selector:@selector(takeback:)
     name:MBCTakebackNotification
     object:fDocument];
	[notificationCenter 
     addObserver:self
     selector:@selector(endOfGame:)
     name:MBCGameEndNotification
     object:fDocument];
    fHasObservers = YES;
}

- (void)opponentMoved:(NSNotification *)n
{
    dispatch_async(dispatch_get_main_queue(), ^{
        [fDocument updateMatchForLocalMove];
    });
}

- (void)takeback:(NSNotification *)n
{
}

- (void)endOfGame:(NSNotification *)n
{
    dispatch_async(dispatch_get_main_queue(), ^{
        [fDocument updateMatchForEndOfGame:((MBCMove *)[n userInfo])->fCommand];
    });
}

@end

// Local Variables:
// mode:ObjC
// End:
