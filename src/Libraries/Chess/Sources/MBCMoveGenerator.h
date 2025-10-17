/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#import "MBCBoard.h"

/*
 * An MBCMoveGenerator generates all legal moves from a position (for various
 * variants and various definitions of "legality" and communicates them to 
 * an object of a class derived from MBCMoveBuilder. 
 */
@protocol MBCMoveBuilder 

- (void) startMoveList:(BOOL)white;
- (void) startUnambiguousMoves;
- (void) endMoveList;
- (void) validMove:(MBCPiece)piece from:(MBCSquare)from to:(MBCSquare)to;
- (void) validMove:(MBCPiece)piece from:(MBCSquare)from to:(MBCSquare)to 
	capturing:(MBCPiece)victim;
- (void) validDrop:(MBCPiece)piece at:(MBCSquare)at;
- (void) validCastle:(MBCPiece)king kingSide:(BOOL)kingSide;

@end

//
// An MBCMoveCounter just counts legal moves
//
@interface MBCMoveCounter : NSObject <MBCMoveBuilder> {
	int		fCount;
	bool	fCounting;
}

- (int)count;

@end

//
// An MBCDebugMoveBuilder prints legal moves
//
@interface MBCDebugMoveBuilder : NSObject <MBCMoveBuilder> {
	bool				fUnambiguous;
	NSMutableArray *	fMoves;
	NSMutableArray *	fUnambiguousMoves;
	NSMutableArray *	fDrops;
}

+ (id)debugMoveBuilder;

@end

typedef uint64_t MBCBoardMask;
//
// An MBCPieceMoves collects all legal moves for one piece type
//
struct MBCPieceMoves {
	int				fNumInstances; // Max. 16 (Pawns in crazyhouse)
	MBCSquare		fFrom[16];
	MBCBoardMask	fTo[16];
};
//
// An MBCMoveCollection collects all legal moves
//
struct MBCMoveCollection {
	MBCPieceMoves	fMoves[7];
	MBCPieceMoves	fUnambiguousMoves[7];
	MBCBoardMask	fPawnDrops;
	MBCBoardMask	fPieceDrops;
	char			fDroppablePieces;
	bool			fCastleKingside;
	bool			fCastleQueenside;
	bool			fWhiteMoves;

	void AddMove(bool unambig, MBCPiece piece, MBCSquare from, MBCSquare to);
	void AddDrop(MBCPiece piece, MBCSquare at);
	void AddCastle(bool kingSide);
};

//
// An MBCMoveCollector collects all legal moves in an MBCMoveCollection
//
@interface MBCMoveCollector : NSObject <MBCMoveBuilder> {
	bool				fUnambiguous;
	MBCMoveCollection	fCollection;
};

- (MBCMoveCollection *) collection;

@end

class MBCMoveGenerator {
public:
	MBCMoveGenerator(id <MBCMoveBuilder> builder, MBCVariant variant, long flags);
	void SetVariant(MBCVariant variant);
	void Generate(bool white, const MBCPieces & position);
	void Ambiguities(MBCSquare from, MBCSquare to, const MBCPieces & position);
	bool InCheck(bool white, const MBCPieces & position);
    bool InCheckMate(bool white, const MBCPieces & position);
    bool InStaleMate(bool white, const MBCPieces & position);
private:
	bool	TryMove(MBCPiece piece, MBCSquare from, MBCSquare to);
	bool	TryMove(MBCPiece piece, MBCSquare from, int dCol, int dRow);
	void	TryMoves(MBCPiece piece, MBCSquare from, int dCol, int dRow);
	void	TryMoves(MBCPiece piece, MBCSquare from);
	void    TryCastle();
	void	TryDrops();
	void	TryMoves(bool unambiguous);

	id <MBCMoveBuilder> 	fBuilder;
	long					fFlags;
	MBCVariant				fVariant;
	MBCPieceCode			fColor;
	MBCPiece				fPieceFilter;
	MBCSquare				fTargetFilter;
	const MBCPieces	*		fPosition;
	bool					fUnambiguous;
	uint8_t					fTargetUsed[64];
	uint8_t					fTargetAmbiguous[64];
};

//
// An MBCCheckCounter counts moves that don't leave the king in check
//
@interface MBCCheckCounter : MBCMoveCounter {
    bool                  fWhite;
    bool                  fCanCastle;
    MBCMoveGenerator *    fGenerator;
    MBCPieces             fPosition;
}

- (id)initForWhite:(BOOL)white variant:(MBCVariant)variant position:(const MBCPieces *)pos canCastle:(BOOL)canCastle;

@end

// Local Variables:
// mode:ObjC
// End:
