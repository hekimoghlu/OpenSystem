/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#import <AppKit/AppKit.h>

#import "MBCBoard.h"

#ifdef CHESS_TUNER
#import "Chess_Tuner-Swift.h"
#else
#import "Chess-Swift.h"
#endif

@class MBCBoard;
@class MBCBoardView;
@class MBCEngine;
@class MBCInteractivePlayer;
@class NSSpeechSynthesizer;
@class MBCDocument;

@interface MBCController : NSObject <GKLocalPlayerListener, MBCSharePlayConnectionDelegate>
{
    IBOutlet NSObjectController *   fCurrentDocument;
    
    NSMutableArray *                fMatchesToLoad;
    NSArray *                       fExistingMatches;
    NSMutableDictionary *           fAchievements;
}

@property (nonatomic, assign)   GKLocalPlayer * localPlayer;
@property (nonatomic)           BOOL            logMouse;
@property (nonatomic)           BOOL            dumpLanguageModels;
@property (assign) IBOutlet     NSMenuItem    * sharePlaySessionMenuItem;


- (id) init;
- (void) awakeFromNib;
- (IBAction) newGame:(id)sender;
- (void)startNewOnlineGame:(GKTurnBasedMatch *)match withDocument:(MBCDocument *)doc;
- (void) loadMatch:(NSString *)matchID;
- (void) setValue:(float)value forAchievement:(NSString *)ident;
- (void) updateApplicationBadge;
- (void)stopRecordingForWindow:(NSWindow *)window;
- (NSWindowController *)currentWindowController;

#if HAS_FLOATING_BOARD
- (IBAction) toggleFloating:(id)sender;
#endif

@end

// Local Variables:
// mode:ObjC
// End:
