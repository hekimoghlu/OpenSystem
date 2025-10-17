/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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
//  PMPowerModeAnalytics.h
//  PowerManagement
//
//  Created by Saurabh Shah on 4/26/21.
//

#ifndef PMPowerModeAnalytics_h
#define PMPowerModeAnalytics_h

#import <Foundation/Foundation.h>

#define POWERMODE_ANALYTICS_ON_DEVICE (!TARGET_OS_SIMULATOR && !XCTEST)

@interface PMPowerModeAnalytics : NSObject

+ (void)sendAnalyticsEvent:(NSNumber * _Nonnull)newState
          withBatteryLevel:(NSNumber * _Nonnull)level
                fromSource:(NSString * _Nonnull)source
               withCharger:(NSNumber * _Nonnull)pluggedIn
     withDurationInMinutes:(NSNumber * _Nonnull)duration
                 forStream:(NSString * _Nonnull)stream;

+ (void)sendAnalyticsDaily:(NSNumber * _Nonnull)enabled
                 forStream:(NSString * _Nonnull)stream;

@end

#endif /* PMPowerModeAnalytics_h */
