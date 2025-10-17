/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
//  _PMCoreSmartPowerNap.h
//  LowPowerMode-Embedded
//
//  Created by Prateek Malhotra on 12/7/22.
//

#ifndef _PMCoreSmartPowerNap_h
#define _PMCoreSmartPowerNap_h

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <LowPowerMode/_PMCoreSmartPowerNapProtocol.h>
#import <LowPowerMode/_PMCoreSmartPowerNapCallbackProtocol.h>

#define kIOPMCoreSmartPowerNapNotifyName "com.apple.powerd.coresmartpowernap"
#define kIOPMCoreSmartPowerNapExit  0
#define kIOPMCoreSmartPowerNapTransient 1
#define kIOPMCoreSmartPowerNapEntry 2

extern NSString *const kPMCoreSmartPowerNapServiceName;

@interface _PMCoreSmartPowerNap : NSObject <_PMCoreSmartPowerNapProtocol, _PMCoreSmartPowerNapCallbackProtocol>
@property (nonatomic, retain) NSXPCConnection *connection;
@property (nonatomic, retain) NSString *identifier;
@property (nonatomic, copy) _PMCoreSmartPowerNapCallback callback;
@property (nonatomic, retain) dispatch_queue_t callback_queue;
@property (nonatomic) _PMCoreSmartPowerNapState current_state;
@property BOOL connection_interrupted;
/*
 Register to receive updates when core smart power nap state changes
 */
- (void)registerWithCallback:(dispatch_queue_t)queue callback:(_PMCoreSmartPowerNapCallback)callback;

/*
 Unregister
 */
- (void)unregister;

/*
 Get current state of core smart power nap. States are defined in _PMCoreSmartPowerNapProtocol.h. This
 state is cached in the client
 */
- (_PMCoreSmartPowerNapState)state;

/*
 Get current state of core smart power nap from powerd. This is a blocking synchronous call
 */
- (_PMCoreSmartPowerNapState)syncState;

/*
 Re-register after powerd exits
 */
- (void)reRegister;
@end


#endif /* _PMCoreSmartPowerNap_h */
