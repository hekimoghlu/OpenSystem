/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
//  PMSmartPowerNapPredictor.h
//  PMSmartPowerNapPredictor
//
//  Created by Archana on 10/19/21.
//



#import <Foundation/Foundation.h>
#import <xpc/private.h>
#import <IOKit/pwr_mgt/powermanagement_mig.h>

#if !XCTEST && !TARGET_OS_BRIDGE
#import <CoreMotion/CMMotionAlarmManager.h>
#import <CoreMotion/CMMotionAlarmDelegateProtocol.h>
@interface PMSmartPowerNapPredictor : NSObject <CMMotionAlarmDelegateProtocol>
#else
@interface PMSmartPowerNapPredictor : NSObject
#endif

+ (instancetype)sharedInstance;

- (instancetype)initWithQueue:(dispatch_queue_t)queue;
- (void)evaluateSmartPowerNap:(BOOL)useractive;
- (void)queryModelAndEngage;
- (void)enterSmartPowerNap;
- (void)exitSmartPowerNapWithReason:(NSString *)reason;
- (void)logEndOfSessionWithReason:(NSString *)reason;
- (void)handleRemoteDeviceIsNear;
- (void)updateSmartPowerNapState:(BOOL)active;
- (void)updateLockState:(uint64_t)state;
- (void)updateBacklightState:(BOOL)state;
- (void)updatePluginState:(BOOL)state;
- (void)updateMotionState:(BOOL)state;
- (void)updateAODEnabledStatus:(BOOL)status;
- (void)updateAmbientState:(BOOL)state;

/*
 Update parameters through pmtool
 */
- (void)updateReentryCoolOffPeriod:(uint32_t)seconds;
- (void)updateReentryDelaySeconds:(uint32_t)seconds;
- (void)updateRequeryDelta:(uint32_t)seconds;
- (void)updateMotionAlarmThreshold:(uint32_t)seconds;
- (void)updateMotionAlarmStartThreshold:(uint32_t)seconds;
/*
 saving defaults
 */
- (void)restoreState;
- (void)saveState:(BOOL)enabled withEndTime:(NSDate *)endTime;
- (void)saveInterruptions;
- (BOOL)readStateFromDefaults;
- (NSDate *)readEndTimeFromDefaults;
- (void)updateInterruptionsFromDefaults;
@end


@interface PMSmartPowerNapInterruption : NSObject
@property (retain) NSDate *start;
@property (retain) NSDate *end;
@property BOOL is_transient;

-(instancetype)initWithStart:(NSDate *)date;
@end

void setSPNRequeryDelta(xpc_object_t remoteConnection, xpc_object_t msg);
