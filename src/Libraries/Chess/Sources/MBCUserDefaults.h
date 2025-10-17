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
#import <Foundation/Foundation.h>

extern NSString * const kMBCBoardStyle;
extern NSString * const kMBCListenForMoves;
extern NSString * const kMBCPieceStyle;
extern NSString * const kMBCNewGamePlayers;
extern NSString * const kMBCNewGameVariant;
extern NSString * const kMBCNewGameSides;
extern NSString * const kMBCSearchTime;
extern NSString * const kMBCMinSearchTime;
extern NSString * const kMBCSpeakMoves;
extern NSString * const kMBCSpeakHumanMoves;
extern NSString * const kMBCDefaultVoice;
extern NSString * const kMBCAlternateVoice;
extern NSString * const kMBCGameCity;
extern NSString * const kMBCGameCountry;
extern NSString * const kMBCGameEvent;
extern NSString * const kMBCHumanName;
extern NSString * const kMBCHumanName2;
extern NSString * const kMBCBattleScars;
extern NSString * const kMBCBoardAngle;
extern NSString * const kMBCBoardSpin;
extern NSString * const kMBCCastleSides;
extern NSString * const kMBCGCVictories;
extern NSString * const kMBCShowGameLog;
extern NSString * const kMBCShowEdgeNotation;

extern NSString * const kMBCShareplayEnabledFF;
extern NSString * const kMBCUseMetalRendererFF;

@interface MBCUserDefaults : NSObject

/*!
 @abstract isSharePlayEnabled
 @return BOOL value indicating whether or not SharePlay is enabled.
 @discussion This function reads the BOOL value for key "SharePlayEnabled" from the Defaults.plist in the bundle.
*/
+ (BOOL)isSharePlayEnabled;

/*!
 @abstract isMetalRenderingEnabled
 @return BOOL value indicating whether or not to render with Metal (YES), or OpenGL (NO)
 @discussion This function returns whether or not to use Metal rendering. Default NO.
*/
+ (BOOL)isMetalRenderingEnabled;

/*!
 @abstract usingScreenCaptureKit
 @return BOOL value indicating whether or not enabled SCK screen recording
 @discussion This function returns whether or not to use SCK recording. Default NO.
*/
+ (BOOL)usingScreenCaptureKit;

@end
