/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
//  _PMSmartPowerNap.h
//  _PMSmartPowerNap
//
//  Created by Archana on 9/14/21.
//

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import <LowPowerMode/_PMSmartPowerNapProtocol.h>
#import <LowPowerMode/_PMSmartPowerNapCallbackProtocol.h>

#define kIOPMSmartPowerNapNotifyName "com.apple.powerd.smartpowernap"
#define kIOPMSmartPowerNapExit  0
#define kIOPMSmartPowerNapTransient 1
#define kIOPMSmartPowerNapEntry 2
#define kIOPMSmartPowerNapInterruptionNotifyName "com.apple.powerd.smartpowernap.interruption"

extern NSString *const kPMSmartPowerNapServiceName;

@interface _PMSmartPowerNap : NSObject <_PMSmartPowerNapProtocol, _PMSmartPowerNapCallbackProtocol>
@property (nonatomic, retain) NSXPCConnection *connection;
@property (nonatomic, retain) NSString *identifier;
@property (nonatomic, copy) _PMSmartPowerNapCallback callback;
@property (nonatomic, retain) dispatch_queue_t callback_queue;
@property (nonatomic) _PMSmartPowerNapState current_state;
@property BOOL connection_interrupted;
/*
 Register to receive updates when smart power nap state changes
 */
- (void)registerWithCallback:(dispatch_queue_t)queue callback:(_PMSmartPowerNapCallback)callback;

/*
 Unregister
 */
- (void)unregister;

/*
 Get current state of smart power nap. States are defined in _PMSmartPowerNapProtocol.h. This
 state is cached in the client
 */
- (_PMSmartPowerNapState)state;

/*
 Get current state of smart power nap from powerd. This is a blocking synchronous call
 */
- (_PMSmartPowerNapState)syncState;

/*
 Re-register after powerd exits
 */
- (void)reRegister;
@end

