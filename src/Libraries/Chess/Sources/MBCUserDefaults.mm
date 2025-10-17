/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
#import <Foundation/Foundation.h>

#import "MBCUserDefaults.h"


NSString * const kMBCBoardStyle         = @"MBCBoardStyle";
NSString * const kMBCListenForMoves     = @"MBCListenForMoves";
NSString * const kMBCPieceStyle         = @"MBCPieceStyle";
NSString * const kMBCNewGamePlayers     = @"MBCNewGamePlayers";
NSString * const kMBCNewGameVariant     = @"MBCNewGameVariant";
NSString * const kMBCNewGameSides       = @"MBCNewGameSides";
NSString * const kMBCSearchTime         = @"MBCSearchTime";
NSString * const kMBCMinSearchTime      = @"MBCMinSearchTime";
NSString * const kMBCSpeakMoves         = @"MBCSpeakMoves";
NSString * const kMBCSpeakHumanMoves    = @"MBCSpeakHumanMoves";
NSString * const kMBCDefaultVoice       = @"MBCDefaultVoice";
NSString * const kMBCAlternateVoice     = @"MBCAlternateVoice";
NSString * const kMBCGameCity           = @"MBCGameCity";
NSString * const kMBCGameCountry        = @"MBCGameCountry";
NSString * const kMBCGameEvent          = @"MBCGameEvent";
NSString * const kMBCHumanName          = @"MBCHumanName";
NSString * const kMBCHumanName2         = @"MBCHumanName2";
NSString * const kMBCBattleScars        = @"MBCBattleScars";
NSString * const kMBCBoardAngle         = @"MBCBoardAngle";
NSString * const kMBCBoardSpin          = @"MBCBoardSpin";
NSString * const kMBCCastleSides        = @"MBCCastleSides";
NSString * const kMBCGCVictories        = @"MBCGCVictories";
NSString * const kMBCShowGameLog        = @"MBCShowGameLog";
NSString * const kMBCShowEdgeNotation   = @"MBCShowEdgeNotation";
NSString * const kMBCSharePlayEnabledFF = @"SharePlayEnabled";
NSString * const kMBCUseMetalRendererFF = @"UseMetalRenderer";

@implementation MBCUserDefaults

+ (BOOL)isSharePlayEnabled {
    static BOOL sIsEnabled;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSString *path = [[NSBundle mainBundle] pathForResource:@"Defaults" ofType:@"plist"];
        NSDictionary *dict = [[NSDictionary alloc] initWithContentsOfFile:path];
        sIsEnabled = [[dict objectForKey:kMBCSharePlayEnabledFF] boolValue];
    });
    return sIsEnabled;
}

+ (BOOL)isMetalRenderingEnabled {
    static BOOL sUsingMetal;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sUsingMetal = NO;
    });
    return sUsingMetal;
}

+ (BOOL)usingScreenCaptureKit {
    static BOOL sUsingSCK;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sUsingSCK = NO;
    });
    return sUsingSCK;
}

@end
