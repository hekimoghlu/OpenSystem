/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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
//  _PMLowPowerMode.h
//
//  Created by Andrei Dorofeev on 1/14/15.
//  Copyright Â© 2015,2020 Apple Inc. All rights reserved.
//

#import <TargetConditionals.h>
#import <AppleFeatures/AppleFeatures.h>
#import <LowPowerMode/_PMLowPowerModeProtocol.h>
#import <LowPowerMode/_PMPowerModeState.h>

extern NSString *const kPMLowPowerModeServiceName;

extern NSString *const kPMLPMSourceSpringBoardAlert;
extern NSString *const kPMLPMSourceReenableBulletin;
extern NSString *const kPMLPMSourceControlCenter;
extern NSString *const kPMLPMSourceSettings;
extern NSString *const kPMLPMSourceSiri;
extern NSString *const kPMLPMSourceLostMode;
extern NSString *const kPMLPMSourceSystemDisable;
extern NSString *const kPMLPMSourceWorkouts;


@interface _PMLowPowerMode : NSObject <_PMLowPowerModeProtocol>

+ (instancetype)sharedInstance;

// Synchronous flavor. The one from Protocol is async.
- (BOOL)setPowerMode:(PMPowerMode)mode fromSource:(NSString *)source;
- (BOOL)setPowerMode:(PMPowerMode)mode fromSource:(NSString *)source withParams:(NSDictionary *)params;
- (PMPowerMode)getPowerMode;

@end
