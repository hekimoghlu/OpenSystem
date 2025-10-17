/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
//  _PMCoreSmartPowerNapProtocol.h
//  LowPowerMode-Embedded
//
//  Created by Prateek Malhotra on 12/7/22.
//

#ifndef _PMCoreSmartPowerNapProtocol_h
#define _PMCoreSmartPowerNapProtocol_h

// states
typedef NS_ENUM(uint8_t, _PMCoreSmartPowerNapState) {
    _PMCoreSmartPowerNapStateOff = 0,
    _PMCoreSmartPowerNapStateOn,
};

// callback provided by clients for state udpates
typedef void(^_PMCoreSmartPowerNapCallback)(_PMCoreSmartPowerNapState state);

@protocol _PMCoreSmartPowerNapProtocol

// register clients with powerd
- (void)registerWithIdentifier: (NSString *)identifier;

// unregister
- (void)unregisterWithIdentifier: (NSString *)identifier;

// only to be used by testing tools. Entitlement will be enforced
- (void)setState:(_PMCoreSmartPowerNapState)state;

- (void)setCSPNQueryDelta:(uint32_t)seconds;

- (void)setCSPNRequeryDelta:(uint32_t)seconds;

- (void)setCSPNIgnoreRemoteClient:(uint32_t)state;

- (void)setCSPNMotionAlarmThreshold:(uint32_t)seconds;

- (void)setCSPNMotionAlarmStartThreshold:(uint32_t)seconds;

/*
 Get current state of core smart power nap from powerd.
 */
- (void)syncStateWithHandler:(_PMCoreSmartPowerNapCallback)handler;
@end



#endif /* _PMCoreSmartPowerNapProtocol_h */
