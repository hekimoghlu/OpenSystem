/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#ifndef SECURITY_OT_OTJOININGCONFIGURATION_H
#define SECURITY_OT_OTJOININGCONFIGURATION_H 1

#if __OBJC2__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OTJoiningConfiguration : NSObject <NSSecureCoding>

@property (nonatomic, strong) NSString* protocolType;
@property (nonatomic, strong) NSString* uniqueDeviceID;
@property (nonatomic, strong) NSString* uniqueClientID;
@property (nonatomic, strong) NSString* pairingUUID;
@property (nonatomic) uint64_t epoch;
@property (nonatomic) BOOL isInitiator;
@property (nonatomic) BOOL testsEnabled;

// Set this to non-zero if you want to configure your timeouts
@property int64_t timeout;

- (instancetype)initWithProtocolType:(NSString*)protocolType
                      uniqueDeviceID:(NSString*)uniqueDeviceID
                      uniqueClientID:(NSString*)uniqueClientID
                         pairingUUID:(NSString* _Nullable)pairingUUID
                               epoch:(uint64_t)epoch
                         isInitiator:(BOOL)isInitiator;
-(instancetype)init NS_UNAVAILABLE;
- (void)enableForTests;

@end
NS_ASSUME_NONNULL_END

#endif /* __OBJC2__ */
#endif /* SECURITY_OT_OTJOININGCONFIGURATION_H */
