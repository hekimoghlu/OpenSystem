/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
//  PMCoreSmartPowerNapPredictor_Testing.h
//  PowerManagement
//
//  Created by Prateek Malhotra on 12/7/22.
//

#ifndef PMCoreSmartPowerNapPredictor_Testing_h
#define PMCoreSmartPowerNapPredictor_Testing_h

#import <Foundation/Foundation.h>
#import "PMCoreSmartPowerNapPredictor.h"

@interface PMCoreSmartPowerNapPredictor(Testing)
@property BOOL feature_enabled;
@property (readonly) BOOL in_coresmartpowernap;
@property (readonly) BOOL session_interrupted;
@property (readonly) BOOL should_reenter;
@property BOOL current_useractive;
@property BOOL skipEndOfSessionTimer;
@property int max_interruptions;
@property double max_interruption_duration;
@property int interruption_cooloff;
@property double interruption_session_duration;
@property NSDate *interruption_session_start;
@property NSDate *full_session_start_time;
@property NSDate *cumulative_interruption_session_start;
@property double cumulative_interruption_session_duration;
@property NSDate *predicted_end_time;


- (void)logNotEngaging;
- (void)initializeTrialClient;
- (void)updateTrialFactors;
@end



#endif /* PMCoreSmartPowerNapPredictor_Testing_h */
