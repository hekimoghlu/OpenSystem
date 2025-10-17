/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#pragma once

#if USE(APPLE_INTERNAL_SDK) || !PLATFORM(APPLETV)
#import <CoreMotion/CoreMotion.h>
#else

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN


typedef struct {
    double x;
    double y;
    double z;
} CMAcceleration;

typedef struct {
    double x;
    double y;
    double z;
} CMRotationRate;


@interface CMLogItem : NSObject <NSSecureCoding, NSCopying>
@end


@interface CMAttitude : NSObject <NSCopying, NSSecureCoding>
@property (readonly, nonatomic) double roll;
@property (readonly, nonatomic) double pitch;
@property (readonly, nonatomic) double yaw;
@end


@interface CMDeviceMotion : CMLogItem
@property (readonly, nonatomic) CMAttitude *attitude;
@property (readonly, nonatomic) CMRotationRate rotationRate;
@property (readonly, nonatomic) CMAcceleration gravity;
@property (readonly, nonatomic) CMAcceleration userAcceleration;
@end


@interface CMAccelerometerData : CMLogItem
@property (readonly, nonatomic) CMAcceleration acceleration;
@end


@interface CMMotionManager : NSObject
@property (assign, nonatomic) NSTimeInterval accelerometerUpdateInterval;
@property (readonly, nullable) CMAccelerometerData *accelerometerData;

- (void)startAccelerometerUpdates;
- (void)stopAccelerometerUpdates;

@property (assign, nonatomic) NSTimeInterval deviceMotionUpdateInterval;
@property (readonly, nonatomic, getter=isDeviceMotionAvailable) BOOL deviceMotionAvailable;
@property (readonly, nullable) CMDeviceMotion *deviceMotion;

- (void)startDeviceMotionUpdates;
- (void)stopDeviceMotionUpdates;
@end

NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK) || !PLATFORM(APPLETV)
