/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#import "MBCBoardEnums.h"
#import <OpenGL/gl.h>
#import <Cocoa/Cocoa.h>
#import <stdio.h>

extern NSString *   gVariantName[];
extern const char   gVariantChar[];

extern const MBCSide gHumanSide[];
extern const MBCSide gEngineSide[];

//
// MBCMove - A move
//
@interface MBCMove : NSObject
{
@public
    MBCMoveCode		fCommand;		// Command
    MBCSquare		fFromSquare;	// Starting square of piece if move
    MBCSquare		fToSquare;		// Finishing square if move or drop
    MBCPiece		fPiece;			// Moved or dropped piece
    MBCPiece		fPromotion;		// Pawn promotion piece
    MBCPiece		fVictim;		// Captured piece, set by [board makeMove]
    MBCCastling		fCastling;		// Castling move, set by [board makeMove]
    BOOL			fEnPassant;		// En passant, set by [board makeMove]
    BOOL           fCheck;        // Check, set by [board makeMove]
    BOOL           fCheckMate;    // Checkmate, set asynchronously
    BOOL 			fAnimate;		// Animate on board
}

- (id) initWithCommand:(MBCMoveCode)command;
+ (id) newWithCommand:(MBCMoveCode)command;
+ (id) moveWithCommand:(MBCMoveCode)command;
- (id) initFromCompactMove:(MBCCompactMove)move;
+ (id) newFromCompactMove:(MBCCompactMove)move;
+ (id) moveFromCompactMove:(MBCCompactMove)move;
- (id) initFromEngineMove:(NSString *)engineMove;
+ (id) newFromEngineMove:(NSString *)engineMove;
+ (id) moveFromEngineMove:(NSString *)engineMove;
+ (BOOL) compactMoveIsWin:(MBCCompactMove)move;

- (NSString *) localizedText;
- (NSString *) engineMove;
- (NSString *) origin;
- (NSString *) operation;
- (NSString *) destination;
- (NSString *) check;

@end

//
// MBCPieces - The full position representation
//
struct MBCPieces {
	MBCPiece		fBoard[64];
	char			fInHand[16];
	MBCSquare		fEnPassant;	// Current en passant square, if any
    
    bool            NoPieces(MBCPieceCode color);
};

//
// MBCBoard - The game board
//
@interface MBCBoard : NSObject
{
	MBCPieces			fCurPos;
	MBCPieces			fPrvPos;
	int					fMoveClock;
	MBCVariant			fVariant;
	NSMutableArray *	fMoves;
	MBCPiece			fPromotion[2];
    NSMutableArray *    fObservers;
    id                  fDocument;
}

- (void)        removeChessObservers;
- (void) 		setDocument:(id)doc;
- (void)		startGame:(MBCVariant)variant;
- (MBCPiece)	curContents:(MBCSquare)square;	// Read contents of a square
- (MBCPiece)	oldContents:(MBCSquare)square;	// Read contents of a square
- (int)			curInHand:(MBCPiece)piece;		// # of pieces to drop
- (int)			oldInHand:(MBCPiece)piece;		// # of pieces to drop
- (void) 		makeMove:(MBCMove *)move; 		// Move pieces and record
- (MBCCastling) tryCastling:(MBCMove *)move;
- (void)		tryPromotion:(MBCMove *)move;
- (MBCSide)    sideOfMove:(MBCMove *)move;
- (MBCUnique) 	disambiguateMove:(MBCMove *)move;
- (bool) 		undoMoves:(int)numMoves;
- (void) 		commitMove;						// Save position
- (NSString *)	fen;							// Position in FEN notation
- (NSString *)	holding;                        // Pieces held
- (NSString *) 	moves;							// Moves in engine format
- (void)        setFen:(NSString *)fen holding:(NSString *)holding 
				moves:(NSString *)moves;
- (BOOL)		saveMovesTo:(FILE *)f;
- (BOOL) 		canPromote:(MBCSide)side;
- (BOOL)	    canUndo;
- (MBCMove *)   lastMove;
- (int) 		numMoves;
- (MBCMove *)   move:(int)index;
- (MBCPieces *) curPos;
- (MBCPiece)	defaultPromotion:(BOOL)white;
- (void)		setDefaultPromotion:(MBCPiece)piece for:(BOOL)white;
- (MBCMoveCode)outcome;
- (NSString *) stringFromMove:(MBCMove *)move withLocalization:(NSDictionary *)localization;
- (NSString *) extStringFromMove:(MBCMove *)move withLocalization:(NSDictionary *)localization;

@end

NSString * LocalizedString(NSDictionary * localization, NSString * key, NSString * fallback);

#define LOC(key, fallback) LocalizedString(localization, key, fallback)


// Local Variables:
// mode:ObjC
// End:
