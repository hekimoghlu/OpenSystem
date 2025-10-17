/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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

NS_ASSUME_NONNULL_BEGIN

@protocol OTDeviceInformationNameUpdateListener
- (void)deviceNameUpdated;
@end

@protocol OTDeviceInformationAdapter

- (void)setOverriddenMachineID:(NSString* _Nullable)machineID;
- (NSString* _Nullable)getOverriddenMachineID;
- (BOOL)isMachineIDOverridden;
- (void)clearOverride;

/* Returns a string like "iPhone3,5" */
- (NSString*)modelID;

/* Returns the user-entered name for this device */
- (NSString* _Nullable)deviceName;

/* Returns a string describing the current os version */
- (NSString*)osVersion;

/* Returns the serial number as a string. */
- (NSString* _Nullable)serialNumber;

/* register for deviceName updates */
- (void)registerForDeviceNameUpdates:(id<OTDeviceInformationNameUpdateListener>)listener;

/* Returns whether the current device is a homepod */
- (BOOL)isHomePod;

/* Returns whether the current device is a watch */
- (BOOL)isWatch;

/* Returns whether the current device is an AppleTV */
- (BOOL)isAppleTV;

@end

@interface OTDeviceInformationActualAdapter : NSObject <OTDeviceInformationAdapter>

@end

NS_ASSUME_NONNULL_END
