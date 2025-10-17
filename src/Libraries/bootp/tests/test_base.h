/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#ifndef test_base_h
#define test_base_h

#import "test_utils.h"

@interface IPConfigurationFrameworkTestBase : NSObject

@property (nonatomic, strong) NSString * description;
@property (nonatomic, strong) __attribute__((NSObject)) SCDynamicStoreRef store;
@property (nonatomic, strong) dispatch_queue_t storeQueue;
@property (atomic, strong) dispatch_semaphore_t storeSem;
@property (nonatomic, strong) __attribute__((NSObject)) IOEthernetControllerRef interfaceController;
@property (nonatomic, strong) dispatch_queue_t interfaceQueue;
@property (atomic, strong) dispatch_semaphore_t interfaceSem;
@property (nonatomic, strong) NSString * ifname;
@property (nonatomic, weak) NSString * serviceKey;
@property (atomic, strong) dispatch_semaphore_t serviceSem;
@property (nonatomic, weak) NSString * serviceKey2;
@property (atomic, strong) dispatch_semaphore_t serviceSem2;
@property (nonatomic) BOOL alternativeValidation;
@property (nonatomic, weak) NSString * pdServiceKey;
@property (nonatomic, strong) NSString * dhcpServerIfname;
@property (nonatomic, strong) NSString * dhcpClientIfname;

+ (instancetype)sharedInstance;
- (instancetype)init;
- (void)dealloc;
- (BOOL)dynamicStoreInitialize;
- (void)dynamicStoreDestroy;
- (BOOL)ioUserEthernetInterfaceCreate;
- (void)ioUserEthernetInterfaceDestroy;
- (void)setService:(IPConfigurationServiceRef)service;
- (void)setService2:(IPConfigurationServiceRef)service;

@end

#endif /* test_base_h */
