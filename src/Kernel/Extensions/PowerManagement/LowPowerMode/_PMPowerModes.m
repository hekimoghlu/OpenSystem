/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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

//
//  _PMPowerModes.m
//
//  Created by Prateek Malhotra on 6/24/24.
//  Copyright Â© 2024 Apple Inc. All rights reserved.
//

#import "_PMPowerModesProtocol.h"
#import "_PMPowerModes.h"

#if TARGET_OS_OSX

@implementation _PMPowerModes

+ (instancetype)sharedInstance
{
    static dispatch_once_t onceToken;
    static _PMPowerModes *saver = nil;
    dispatch_once(&onceToken, ^{
        saver = [[_PMPowerModes alloc] init];
    });
    return saver;
}

- (IOReturn)registerForUpdatesOfPowerMode:(PMPowerMode)mode 
                           withIdentifier:(NSString *)clientIdentifier
                             withCallback:(PMPowerModeUpdateHandler)handler
{
    return kIOReturnUnsupported;
}

- (IOReturn)registerForUpdatesWithIdentifier:(NSString *)clientIdentifier 
                                withCallback:(PMPowerModeUpdateHandler)handler
{
    return kIOReturnUnsupported;
}

- (IOReturn)setStateTo:(PMPowerModeState)newState
          forPowerMode:(PMPowerMode)mode
            fromSource:(NSString *)source
   withExpirationEvent:(PMPowerModeExpirationEventType)expirationEvent
             andParams:(PMPowerModeExpirationEventParams)expirationParams
          withCallback:(PMPowerModeUpdateHandler)handler
{
    return kIOReturnUnsupported;
}

- (BOOL)supportsPowerMode:(PMPowerMode)mode
{
    if (mode == PMNormalPowerMode) {
        return YES;
    }
    return NO;
}

- (BOOL)supportsPowerModeSelectionUI
{
    return NO;
}

- (PMPowerMode)currentPowerMode {
    return PMNormalPowerMode;
}

- (_PMPowerModeSession *)currentPowerModeSession {
    return [[_PMPowerModeSession alloc] init];
}

@end

#endif
