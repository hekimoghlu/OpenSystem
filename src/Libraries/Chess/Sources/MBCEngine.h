/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#import "MBCPlayer.h"

//
// MBCEngine is an instance of MBCPlayer, but it also serves other
// purposes like move generation and checking.
//
@interface MBCEngine : MBCPlayer <NSPortDelegate>
{
    NSTask * 		fEngineTask;	// The chess engine
    NSFileHandle * 	fToEngine;		// Writing to the engine
    NSFileHandle * 	fFromEngine;	// Reading from the engine
    NSPipe * 		fToEnginePipe;
    NSPipe * 		fFromEnginePipe;
    NSRunLoop * 	fMainRunLoop;	
    NSPort * 		fEngineMoves;	// Moves parsed from engine
    NSPortMessage * fMove;			// 	... the move
    MBCMove * 		fLastMove;		// Last move played by player
    MBCMove * 		fLastPonder;	// Last move pondered by engine
    MBCMove * 		fLastEngineMove;// Last move played by engine
    MBCSide 		fLastSide;		// Side of player
    bool 			fThinking;		// Engine currently thinking
    bool 			fWaitForStart;	// Wait for StartGame command
    bool 			fSetPosition;	// Position set up already
    bool 			fTakeback;		// Pending takeback
    bool 			fEngineEnabled;	// Engine moves enabled?
    bool 			fNeedsGo;		// Engine needs explicit start
    MBCSide 		fSide;			// What side(s) engine is playing
    NSTimeInterval  fDontMoveBefore;// Delay next engine move
    bool            fIsLogging;
    NSFileHandle *  fEngineLogFile;
    BOOL            fHasObservers;
}

- (id) init;
- (BOOL) isLogging;
- (void) setLogging:(BOOL)logging;
- (void) shutdown;
- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay;
- (void) setSearchTime:(int)time;
+ (int) secondsForTime:(int)time;
- (MBCMove *) lastPonder;
- (MBCMove *) lastEngineMove;
- (void) setGame:(MBCVariant)variant fen:(NSString *)fen holding:(NSString *)holding moves:(NSString *)moves;
- (void) takeback;

//
// Hooked up internally
//
- (void) opponentMoved:(NSNotification *)notification;

//
// Used internally
//
- (void) runEngine:(id)sender;
- (void) enableEngineMoves:(BOOL)enable;
- (void) handlePortMessage:(NSPortMessage *)message;
- (void) writeToEngine:(NSString *)string;
- (void) interruptEngine;
- (void) flipSide;
- (NSString *) notificationForSide;
- (NSString *) squareToCoord:(MBCSquare)square;

@end

// Local Variables:
// mode:ObjC
// End:
