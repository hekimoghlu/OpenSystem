/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 5, 2024.
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
//  PMSmartPowerNapService.h
//  PMSmartPowerNapService
//
//  Created by Archana on 9/22/21.
//

#ifndef PMSmartPowerNapService_h
#define PMSmartPowerNapService_h
#import <Foundation/Foundation.h>
#import <Foundation/NSXPCConnection.h>
#if !XCTEST
#import <BacklightServices/BLSBacklightChangeRequest.h>
#import <BacklightServices/BLSBacklightStateObserving.h>
#import <BacklightServices/BLSBacklightChangeEvent.h>

#import <BacklightServices/BLSBacklight.h>
#endif
#import "_PMSmartPowerNapProtocol.h"


#if !XCTEST
@interface PMSmartPowerNapService : NSXPCListener <NSXPCListenerDelegate, _PMSmartPowerNapProtocol, BLSBacklightStateObserving>
#else
@interface PMSmartPowerNapService : NSXPCListener <NSXPCListenerDelegate, _PMSmartPowerNapProtocol>
#endif

+ (instancetype)sharedInstance;
- (void)updateAmbientState:(BOOL)state;
- (void)updateState:(_PMSmartPowerNapState)state;
- (void)enterSmartPowerNap;
- (void)exitSmartPowerNap;
@end
#endif /* PMSmartPowerNapService_h */
