/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
//  _PMPowerModeSession.h
//
//  Created by Prateek Malhotra on 6/24/24.
//  Copyright Â© 2024 Apple Inc. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <LowPowerMode/_PMPowerModeState.h>

#if TARGET_OS_OSX

typedef NS_ENUM(NSInteger, PMPowerModeExpirationEventType) {
    PMPowerModeExpirationEventTypeNone = 0,                                       // Indefinite sessions
    PMPowerModeExpirationEventTypeTime = 1,
    PMPowerModeExpirationEventTypeSufficientCharge = 2,                           // LPM only
    PMPowerModeExpirationEventTypePowerSourceChange = 3                           // Portables only
};

typedef NS_ENUM(NSInteger, PMPowerModeExpirationEventParams) {
    PMPowerModeExpirationEventParamsNone = 0,
    PMPowerModeExpirationEventParamsTime_1hour = 1,
    PMPowerModeExpirationEventParamsTime_UntilTomorrow = 2
};

@interface _PMPowerModeSession : NSObject

/**
 * @abstract            The mode that is active for this session
*/
@property (nonatomic, readonly) PMPowerMode mode;

/**
 * @abstract            The state of the power mode
*/
@property (nonatomic, readonly) PMPowerModeState state;

/**
 * @abstract            Expiration type for the active power mode
 * @discussion          Depending on the expiration type, other properties may be useful to inspect.
 *                      For example, PMPowerModeExpirationReasonTime will have the `expiresAt` set.
*/
@property (nonatomic, readonly) PMPowerModeExpirationEventType expirationEventType;

/**
 * @abstract            Start time for this session
*/
@property (nonatomic, readonly) NSDate *startedAt;

/**
 * @abstract            When the session is expected to expire.
 * @discussion          Will only be set for sessions of expirationReason PMPowerModeExpirationReasonTime
*/
@property (nonatomic, readonly) NSDate *expiresAt;

@end


#endif
