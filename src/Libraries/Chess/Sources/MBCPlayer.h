/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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

#import "MBCBoard.h"

//
// Moves are sent to all interested parties. Trusted clients,
// e.g., chess engines, broadcast MBC*MoveNotification. Untrusted
// clients, e.g. human players, broadcast MBCUnchecked*MoveNotification,
// which is checked by the engine and either turned into a
// MBC*MoveNotification or a MBCIllegalMoveNotification. The legal move
// notification is executed by the board view (possibly using animation)
// and turned into a MBCEndMoveNotification. When the user takes back a move,
// MBCTakebackNotification is broadcast.
//
extern NSString * const MBCGameLoadNotification;
extern NSString * const MBCGameStartNotification;
extern NSString * const MBCWhiteMoveNotification;
extern NSString * const MBCBlackMoveNotification;
extern NSString * const MBCUncheckedWhiteMoveNotification;
extern NSString * const MBCUncheckedBlackMoveNotification;
extern NSString * const MBCIllegalMoveNotification;
extern NSString * const MBCEndMoveNotification;
extern NSString * const MBCTakebackNotification;
extern NSString * const MBCGameEndNotification;

extern NSString * const kMBCHumanPlayer;
extern NSString * const kMBCEnginePlayer;

@class MBCDocument;

//
// MBCPlayer is an abstract superclass for all possible agents
// that can play a side (a human, a chess engine, a network connection)
//
@interface MBCPlayer : NSObject
{
    MBCDocument *   fDocument;
}

@property (nonatomic,assign) MBCDocument *  document;

//
// Start a game playing the black side, the white side, both sides,
// or neither side (in observation mode).
// 
- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay;
- (void) removeChessObservers;

@end

// Local Variables:
// mode:ObjC
// End:
