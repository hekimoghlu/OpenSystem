/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#import "MBCBoard.h"

@interface MBCDocument : NSDocument
{
    MBCBoard *              board;
    MBCVariant              variant;
    MBCPlayers              players;
    NSMutableDictionary *   properties;
    BOOL                    localWhite;
    BOOL                    disallowSubstitutes;
}

@property (nonatomic,assign)    MBCBoard *              board;
@property (nonatomic,readonly)  MBCVariant              variant;
@property (nonatomic,readonly)  MBCPlayers              players;
@property (nonatomic,retain)    GKTurnBasedMatch *      match;
@property (nonatomic,assign,readonly)  NSMutableDictionary *   properties;
@property (nonatomic)           BOOL                    offerDraw;
@property (nonatomic)           BOOL                    ephemeral;
@property (nonatomic)           BOOL                    needNewGameSheet;
@property (nonatomic)           BOOL                    disallowSubstitutes;
@property (nonatomic,retain)    NSArray *               invitees;

+ (BOOL) processNewMatch:(GKTurnBasedMatch *)match variant:(MBCVariant)variant side:(MBCSideCode)side
                document:(MBCDocument *)doc;
- (id) initWithMatch:(GKTurnBasedMatch *)match game:(NSDictionary *)gameData;
- (id) initForNewGameSheet:(NSArray *)invitees;
- (BOOL) canTakeback;
- (BOOL) boolForKey:(NSString *)key;
- (NSInteger) integerForKey:(NSString *)key;
- (float) floatForKey:(NSString *)key;
- (id) objectForKey:(NSString *)key;
- (void) setObject:(id)value forKey:(NSString *)key;
- (void) updateMatchForLocalMove;
- (void) updateMatchForRemoteMove;
- (void) updateMatchForEndOfGame:(MBCMoveCode)cmd;
- (MBCSide) humanSide;
- (MBCSide) engineSide;
- (MBCSide) remoteSide;
- (void) offerTakeback;
- (void) allowTakeback:(BOOL)allow;
- (void) resign;
- (void) updateSearchTime;
- (BOOL) nontrivialHumanTurn;
- (BOOL) gameDone;
- (BOOL) brandNewGame;
- (NSString *)nonLocalPlayerID;

@end

void MBCAbort(NSString * message, MBCDocument * doc);

// Local Variables:
// mode:ObjC
// End:
