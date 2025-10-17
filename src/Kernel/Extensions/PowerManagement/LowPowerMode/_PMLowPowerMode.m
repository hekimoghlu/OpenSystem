/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
//  _PMLowPowerMode.m
//
//  Created by Andrei Dorofeev on 1/14/15.
//  Copyright Â© 2015,2020 Apple Inc. All rights reserved.
//

#import <TargetConditionals.h>
#import <AppleFeatures/AppleFeatures.h>
#import <Foundation/Foundation.h>
#import <Foundation/NSPrivate.h>
#import <Foundation/NSXPCConnection_Private.h>
#import <dispatch/dispatch.h>

#import "_PMLowPowerMode.h"
#import "_PMLowPowerModeProtocol.h"

#define _PM_XPC_TIMEOUT (15 * NSEC_PER_SEC)

// Service Name
NSString *const kPMLowPowerModeServiceName = @"com.apple.powerd.lowpowermode";

NSString *const kPMLPMSourceSpringBoardAlert = @"SpringBoard";
NSString *const kPMLPMSourceReenableBulletin = @"Reenable";
NSString *const kPMLPMSourceControlCenter = @"ControlCenter";
NSString *const kPMLPMSourceSettings = @"Settings";
NSString *const kPMLPMSourceSiri = @"Siri";
NSString *const kPMLPMSourceLostMode = @"LostMode";
NSString *const kPMLPMSourceSystemDisable = @"SystemDisable";
NSString *const kPMLPMSourceWorkouts = @"Workouts";

// Keys for param dictionary

@interface _PMLowPowerMode () {
    NSXPCConnection *_connection;
}

@end

@implementation _PMLowPowerMode

+ (instancetype)sharedInstance
{
    static dispatch_once_t onceToken;
    static _PMLowPowerMode *saver = nil;
    dispatch_once(&onceToken, ^{
        saver = [[_PMLowPowerMode alloc] init];
    });
    return saver;
}

- (instancetype)init
{
    self = [super init];

    if (self) {
        _connection = [[NSXPCConnection alloc] initWithMachServiceName:kPMLowPowerModeServiceName
                                                               options:NSXPCConnectionPrivileged];
        _connection.remoteObjectInterface = [NSXPCInterface interfaceWithProtocol:@protocol(_PMLowPowerModeProtocol)];
        [_connection resume];
    }

    return self;
}

- (void)dealloc
{
    [_connection invalidate];
}

- (void)setPowerMode:(PMPowerMode)mode
          fromSource:(NSString *)source
      withCompletion:(PMSetPowerModeCompletionHandler)handler
{
    [self setPowerMode:mode fromSource:source withParams:nil withCompletion:handler];
}

- (void)setPowerMode:(PMPowerMode)mode
          fromSource:(NSString *)source
          withParams:(NSDictionary *)params
      withCompletion:(PMSetPowerModeCompletionHandler)handler
{
    [[_connection remoteObjectProxyWithErrorHandler:^(NSError *_Nonnull error) {
        handler(NO, error);
    }] setPowerMode:mode
            fromSource:source
            withParams:params
        withCompletion:handler];
}

- (BOOL)setPowerMode:(PMPowerMode)mode fromSource:(NSString *)source
{
    return [self setPowerMode:mode fromSource:source withParams:nil];
}

- (BOOL)setPowerMode:(PMPowerMode)mode fromSource:(NSString *)source withParams:(NSDictionary *)params
{
    __block BOOL ret = YES;
    [[_connection synchronousRemoteObjectProxyWithErrorHandler:^(NSError *_Nonnull __unused error) {
        ret = NO;
        NSLog(@"synchronous connection failed: %@\n", error);
    }] setPowerMode:mode
            fromSource:source
            withParams:params
        withCompletion:^(BOOL success, NSError *error) {
            if (!success || error) {
                ret = NO;
                NSLog(@"setPowerMode failed: %@\n", error);
            }
            return;
        }];
    return ret;
}

- (PMPowerMode)getPowerMode
{
    BOOL enabled = [[NSProcessInfo processInfo] isLowPowerModeEnabled];
    return (enabled ? PMLowPowerMode : PMNormalPowerMode);
}

@end
