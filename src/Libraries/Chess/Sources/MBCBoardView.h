/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#import "MBCBoardCommon.h"
#import "MBCBoardViewInterface.h"
#import <sys/time.h>

extern MBCPieceCode gInHandOrder[];

@class MBCInteractivePlayer;
@class MBCDrawStyle;
@class MBCBoardWin;

@interface MBCBoardView : NSOpenGLView <MBCBoardViewInterface>
{
    MBCBoardWin *         fController;
    MBCInteractivePlayer *fInteractive;
    MBCBoard *            fBoard;
    MBCSquare             fPickedSquare;
    MBCPiece              fSelectedPiece;
    MBCSquare             fSelectedSquare;
    MBCSquare             fSelectedDest;
    MBCPosition           fSelectedPos;
    MBCPosition           fLastSelectedPos;
    float                 fRawAzimuth;
    NSPoint               fOrigMouse;
    NSPoint               fCurMouse;
    struct timeval		    fLastRedraw;
@public
	float					fAzimuth;
	float					fElevation;
	float					fBoardReflectivity;
	float					fLabelIntensity;
	float					fAmbient;
	MBCDrawStyle	*		fBoardDrawStyle[2];
	MBCDrawStyle	*		fPieceDrawStyle[2];
	MBCDrawStyle 	*		fBorderDrawStyle;
	MBCDrawStyle 	*		fSelectedPieceDrawStyle;
	GLfloat					fLightPos[4];
@private
	GLuint					fNumberTextures[8];
	GLuint					fLetterTextures[8];
	int						fMaxFSAA;
	int						fCurFSAA;
	int						fLastFSAASize;
	bool					fNeedStaticModels;
	bool					fIsPickingFormat;
	bool					fIsFloating;
	bool					fWantMouse;
	bool					fNeedPerspective;
	bool					fInAnimation;
	bool					fInBoardManipulation;
	MBCVariant				fVariant;
	MBCSide					fSide;
	MBCSide					fPromotionSide;
	NSDictionary *			fBoardAttr;
	NSDictionary *			fPieceAttr;
	NSString *				fBoardStyle;
	NSString *				fPieceStyle;
	MBCMove	*				fHintMove;
	MBCMove	*				fLastMove;
	NSCursor *				fHandCursor;
	NSCursor *				fArrowCursor;
	MBCPiece				fLastPieceDrawn;
	char					fKeyBuffer;
	float					fAnisotropy;
	GLint					fNumSamples;
    NSTrackingArea *       fTrackingArea;
}

//
// Properties and methods to provide access to ChessTuner
//
@property (nonatomic, assign) float azimuth;
@property (nonatomic, assign) float elevation;
@property (nonatomic, assign) float boardReflectivity;
@property (nonatomic, assign) float labelIntensity;
@property (nonatomic, assign) float ambient;

- (MBCDrawStyle *)boardDrawStyleAtIndex:(NSUInteger)index;
- (MBCDrawStyle *)pieceDrawStyleAtIndex:(NSUInteger)index;
- (MBCDrawStyle *)borderDrawStyle;
- (MBCDrawStyle *)selectedPieceDrawStyle;

- (void)setLightPosition:(vector_float3)lightPosition;
- (vector_float3)lightPosition;

//
// Basic view routines
//
- (id) initWithFrame:(NSRect)rect;
- (void) awakeFromNib;
- (void) drawRect:(NSRect)rect;

- (void) startGame:(MBCVariant)variant playing:(MBCSide) side;
- (void) drawNow;			// Redraw immediately
- (void) profileDraw;		// Redraw in a tight loop
- (void) needsUpdate;		// Perspective changed
- (void) endGame;     		// Clean up the previous game
- (void) startAnimation;	// Start animation
- (void) animationDone;		// Animation finished

//
// Fall back to less memory hungry graphics format
//
- (void) pickPixelFormat:(BOOL)afterFailure;

//
// Change textures
//
- (void) setStyleForBoard:(NSString *)boardStyle pieces:(NSString *)pieceStyle;

//
// Selection manipulation
//
- (void) selectPiece:(MBCPiece)piece at:(MBCSquare)square;
- (void) selectPiece:(MBCPiece)piece at:(MBCSquare)square to:(MBCSquare)dest;
- (void) moveSelectionTo:(MBCPosition *)position;
- (void) unselectPiece;
- (void) clickPiece;

//
// Show hints and last moves
//
- (void) showMoveAsHint:(MBCMove *)move;
- (void) showMoveAsLast:(MBCMove *)move;
- (void) hideMoves;

//
// Translate between squares and positions
//
- (MBCSquare) 	positionToSquare:(const MBCPosition *)position;
- (MBCSquare) 	positionToSquareOrRegion:(const MBCPosition *)position;
- (MBCPosition)	squareToPosition:(MBCSquare)square;
- (void) snapToSquare:(MBCPosition *)position;
- (MBCSide) 	facing;			// What player are we facing?
- (BOOL) 		facingWhite;	// Are we facing white?

//
// Pass mouse clicks on to interactive player?
//
- (void) wantMouse:(BOOL)wantIt;

@end

// Local Variables:
// mode:ObjC
// End:
