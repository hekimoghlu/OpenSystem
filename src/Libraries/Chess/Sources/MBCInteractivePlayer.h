/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>

#import "MBCPlayer.h"
#import "MBCMoveGenerator.h"

#ifdef CHESS_TUNER
#import "Chess_Tuner-Swift.h"
#else
#import "Chess-Swift.h"
#endif


@class MBCBoardWin;
@class MBCLanguageModel;

//
// MBCInteractivePlayer represents humans playing locally
//
@interface MBCInteractivePlayer : MBCPlayer 
{
	IBOutlet MBCBoardWin *	fController;
	MBCLanguageModel *		fLanguageModel;
	MBCSide					fLastSide;
	MBCSide					fSide;
	MBCVariant				fVariant;
	MBCSquare				fFromSquare;
	bool					fStartingSR;
    bool                    fAnnounceCheck;
    SRRecognitionSystem     fRecSystem;
	SRRecognizer			fRecognizer;
	SRLanguageModel			fModel;
	NSData *				fSpeechHelp;
    BOOL                    fHasObservers;
    BOOL                    fPendingMouseUpdate;
}

- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay;
- (void) updateNeedMouse:(id)arg;
- (void) doUpdateNeedMouse;
- (void) allowedToListen:(BOOL)allowed;

//
// The board view translates coordinates into board squares and handles
// dragging.
//
- (void) startSelection:(MBCSquare)square;
- (void) startSelectionWithoutShare:(MBCSquare)square;
- (void) endSelection:(MBCSquare)square animate:(BOOL)animate;
- (void) endSelectionWithoutShare:(MBCSquare)square animate:(BOOL)animate;

//
// If we recognize a move, we have to broadcast it
//
- (void) recognized:(SRRecognitionResult)result;

//
// Announce hint / last move
//
- (void) announceHint:(MBCMove *) move;
- (void) announceLastMove:(MBCMove *) move;

- (void) removeController;

- (void) setLastSide:(MBCSide) side;

@end

// Local Variables:
// mode:ObjC
// End:
