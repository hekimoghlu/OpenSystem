/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
//  PMCoreSmartPowerNapService.h
//  PowerManagement
//
//  Created by Prateek Malhotra on 12/7/22.
//

#ifndef PMCoreSmartPowerNapService_h
#define PMCoreSmartPowerNapService_h

#import <Foundation/Foundation.h>
#import <Foundation/NSXPCConnection.h>
#import "_PMCoreSmartPowerNapProtocol.h"


@interface PMCoreSmartPowerNapService : NSXPCListener <NSXPCListenerDelegate, _PMCoreSmartPowerNapProtocol>

+ (instancetype)sharedInstance;
- (void)updateState:(_PMCoreSmartPowerNapState)state;
- (void)enterCoreSmartPowerNap;
- (void)exitCoreSmartPowerNap;
#if TARGET_OS_OSX
- (NSDate *)timeForNextEvaluation;
#endif
@end

#endif /* PMCoreSmartPowerNapService_h */
