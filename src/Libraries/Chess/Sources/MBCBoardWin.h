/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
#import <GameKit/GameKit.h>
#import <MetalKit/MTKView.h>
#import <UserNotifications/UserNotifications.h>

#ifdef CHESS_TUNER
#import "Chess_Tuner-Swift.h"
#else
#import "Chess-Swift.h"
#endif

#import "MBCBoard.h"
#import "MBCBoardViewInterface.h"

@class MBCBoard;
@class MBCBoardView;
@class MBCBoardMTLView;
@class MBCEngine;
@class MBCInteractivePlayer;
@class MBCGameInfo;
@class MBCRemotePlayer;
@class MBCAnimation;
@class MBCMetalRenderer;

@interface MBCBoardWin : NSWindowController <NSWindowDelegate,
    GKAchievementViewControllerDelegate,
    GKTurnBasedMatchmakerViewControllerDelegate, GKGameCenterControllerDelegate, MBCSharePlayManagerBoardWindowDelegate, UNUserNotificationCenterDelegate, MTKViewDelegate>
{
    NSMutableArray *                fObservers;
    GKAchievementViewController *   fAchievements;
    MBCAnimation *                  fCurAnimation;
    int                             currentSharePlayMoveStringCount;
    MBCMetalRenderer *              fMetalRenderer;
}

@property (nonatomic, assign) IBOutlet MBCBoardView<MBCBoardViewInterface> *           gameView;
@property (nonatomic, assign) IBOutlet MBCBoardMTLView<MBCBoardViewInterface> *        gameMTLView;
@property (nonatomic, assign) IBOutlet NSView *                 mtlBackingView;
@property (nonatomic, assign) IBOutlet NSPanel *                gameNewSheet;
@property (nonatomic, assign) IBOutlet NSBox *                  logContainer;
@property (nonatomic, assign) IBOutlet NSView *                 logView;
@property (nonatomic, assign) IBOutlet MBCBoard *               board;
@property (nonatomic, assign) IBOutlet MBCEngine *              engine;
@property (nonatomic, assign) IBOutlet MBCInteractivePlayer *   interactive;
@property (nonatomic, assign) IBOutlet MBCGameInfo *            gameInfo;
@property (nonatomic, assign) IBOutlet MBCRemotePlayer *        remote;
@property (nonatomic, assign) IBOutlet NSLayoutConstraint *     logViewRightEdgeConstraint;
@property (nonatomic, assign) IBOutlet GKDialogController *     dialogController;
@property (nonatomic, readonly) NSSpeechSynthesizer *           primarySynth;
@property (nonatomic, readonly) NSSpeechSynthesizer *           alternateSynth;
@property (nonatomic, readonly) NSDictionary *                  primaryLocalization;
@property (nonatomic, readonly) NSDictionary *                  alternateLocalization;
@property (assign) IBOutlet NSMenu *playersPopupMenu;
@property (nonatomic, assign) IBOutlet NSMenu * sharePlayMenu;

+ (BOOL)isRenderingWithMetal;
- (NSView <MBCBoardViewInterface> *)renderView;
- (void) removeChessObservers;
- (IBAction)takeback:(id)sender;
- (void) requestTakeback;
- (void) requestDraw;
- (IBAction)resign:(id)sender;
- (IBAction)showHint:(id)sender;
- (IBAction)showLastMove:(id)sender;
- (IBAction)toggleLogView:(id)sender;
- (IBAction) startNewGame:(id)sender;
- (IBAction) cancelNewGame:(id)sender;
- (IBAction) showAchievements:(id)sender;
- (IBAction) profileDraw:(id)sender;
- (void)adjustLogViewForReusedWindow;
- (void)checkEdgeNotationVisibilityForReusedWindow;
- (BOOL)listenForMoves;
- (NSString *)speakOpponentTitle;
- (BOOL)speakMoves;
- (BOOL)speakHumanMoves;
- (IBAction) updatePlayers:(id)sender;
- (BOOL)hideEngineStrength;
- (BOOL)hideNewGameSides;
- (BOOL)hideSpeakMoves;
- (BOOL)hideSpeakHumanMoves;
- (BOOL)hideEngineProperties;
- (BOOL)hideRemoteProperties;
- (NSString *)engineStrength;
+ (NSSet *) keyPathsForValuesAffectingEngineStrength;
- (IBAction) showPreferences:(id)sender;
- (void)setAngle:(float)angle spin:(float)spin;
- (void)handleRemoteResponse:(NSString *)response;
- (void)endAnimation;
- (BOOL)hideSharePlayProperties;
- (IBAction)toggleEdgeNotation:(id)sender;

@end
