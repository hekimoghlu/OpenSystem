/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
//  KCAESGCMDuplexSession.h
//  Security
//
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface KCAESGCMDuplexSession : NSObject <NSSecureCoding>

// Due to design constraints, this session object is the only thing serialized during piggybacking sessions.
// Therefore, we must add some extra data here, which is not strictly part of a AES GCM session.
@property (retain, nullable) NSString* pairingUUID;
@property (retain, nullable) NSString* altDSID;
@property (retain, nullable) NSString* flowID;
@property (retain, nullable) NSString* deviceSessionID;
@property uint64_t piggybackingVersion;
@property uint64_t epoch;

- (nullable NSData*) encrypt: (NSData*) data error: (NSError**) error;
- (nullable NSData*) decryptAndVerify: (NSData*) data error: (NSError**) error;

+ (nullable instancetype) sessionAsSender: (NSData*) sharedSecret
                                  context: (uint64_t) context;
+ (nullable instancetype) sessionAsReceiver: (NSData*) sharedSecret
                                    context: (uint64_t) context;

- (nullable instancetype) initAsSender: (NSData*) sharedSecret
                               context: (uint64_t) context;
- (nullable instancetype) initAsReceiver: (NSData*) sharedSecret
                                 context: (uint64_t) context;
- (nullable instancetype) initWithSecret: (NSData*) sharedSecret
                                 context: (uint64_t) context
                                      as: (bool) inverted;

- (nullable instancetype)initWithSecret:(NSData*)sharedSecret
                                context:(uint64_t)context
                                     as:(bool) sender
                                altDSID:(NSString* _Nullable)altDSID
                            pairingUUID:(NSString* _Nullable)pairingUUID
                    piggybackingVersion:(uint64_t)piggybackingVersion
                                  epoch:(uint64_t)epoch
            NS_DESIGNATED_INITIALIZER;

- (instancetype) init NS_UNAVAILABLE;


- (void)encodeWithCoder:(NSCoder *)aCoder;
- (nullable instancetype)initWithCoder:(NSCoder *)aDecoder;
+ (BOOL)supportsSecureCoding;

@end

NS_ASSUME_NONNULL_END
