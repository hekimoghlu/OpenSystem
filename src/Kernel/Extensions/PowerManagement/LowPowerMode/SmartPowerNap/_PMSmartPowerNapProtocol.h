/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 10, 2024.
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
//  _PMSmartPowerNapProtocol.h
//  _PMSmartPowerNapProtocol
//
//  Created by Archana on 9/21/21.
//


// states
typedef NS_ENUM(uint8_t, _PMSmartPowerNapState) {
    _PMSmartPowerNapStateOff = 0,
    _PMSmartPowerNapStateOn,
};

// callback provided by clients for state udpates
typedef void(^_PMSmartPowerNapCallback)(_PMSmartPowerNapState state);

@protocol _PMSmartPowerNapProtocol

// register clients with powerd
- (void)registerWithIdentifier: (NSString *)identifier;

// unregister
- (void)unregisterWithIdentifier: (NSString *)identifier;

// only to be used by testing tools. Entitlement will be enforced
- (void)setState:(_PMSmartPowerNapState)state;

- (void)setSPNReentryCoolOffPeriod:(uint32_t)seconds;

- (void)setSPNReentryDelaySeconds:(uint32_t)seconds;

- (void)setSPNRequeryDelta:(uint32_t)seconds;

- (void)setSPNMotionAlarmThreshold:(uint32_t)seconds;

- (void)setSPNMotionAlarmStartThreshold:(uint32_t)seconds;

/*
 Get current state of smart power nap from powerd.
 */
- (void)syncStateWithHandler:(_PMSmartPowerNapCallback)handler;
@end

