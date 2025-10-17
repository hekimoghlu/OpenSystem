/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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
/*
 */

#import "MBCPlayer.h"

NSString * const MBCGameLoadNotification            = @"MBCGameLoad";
NSString * const MBCGameStartNotification			= @"MBCGameStart";
NSString * const MBCWhiteMoveNotification			= @"MBCWhMove";
NSString * const MBCBlackMoveNotification			= @"MBCBlMove";
NSString * const MBCUncheckedWhiteMoveNotification  = @"MBCUWMove";
NSString * const MBCUncheckedBlackMoveNotification	= @"MBCUBMove";
NSString * const MBCIllegalMoveNotification			= @"MBCIlMove";
NSString * const MBCEndMoveNotification				= @"MBCEnMove";
NSString * const MBCTakebackNotification			= @"MBCTakeback";
NSString * const MBCGameEndNotification				= @"MBCGameEnd";

NSString * const kMBCHumanPlayer					= @"human";
NSString * const kMBCEnginePlayer					= @"program";

@implementation MBCPlayer 

@synthesize document = fDocument;

- (void) startGame:(MBCVariant)variant playing:(MBCSide)sideToPlay
{
}

@end

// Local Variables:
// mode:ObjC
// End:
