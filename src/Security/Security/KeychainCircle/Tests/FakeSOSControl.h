/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

#ifndef FakeSOSControl_h
#define FakeSOSControl_h

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <Security/SecKeyPriv.h>
#import <Security/SecItemPriv.h>
#import "keychain/SecureObjectSync/SOSAccount.h"
#include "keychain/SecureObjectSync/SOSAccountPriv.h"
#include "keychain/SecureObjectSync/SOSCircle.h"
#import <KeychainCircle/KeychainCircle.h>
#import "utilities/SecCFWrappers.h"

NS_ASSUME_NONNULL_BEGIN

@interface FCPairingFakeSOSControl : NSObject <SOSControlProtocol>
@property (assign) SecKeyRef accountPrivateKey;
@property (assign) SecKeyRef accountPublicKey;
@property (assign) SecKeyRef deviceKey;
@property (assign) SecKeyRef octagonSigningKey;
@property (assign) SecKeyRef octagonEncryptionKey;
@property (assign) SOSCircleRef circle;
@property (assign) SOSFullPeerInfoRef fullPeerInfo;
@property (assign) bool application;
- (instancetype)initWithRandomAccountKey:(bool)randomAccountKey circle:(SOSCircleRef)circle;
- (void)dealloc;
- (SOSPeerInfoRef)peerInfo;
- (void)signApplicationIfNeeded;
@end

@interface FakeNSXPCConnection : NSObject
- (instancetype) initWithControl:(id<SOSControlProtocol>)control;
- (id)remoteObjectProxyWithErrorHandler:(void(^)(NSError * _Nonnull error))failureHandler;
@end

@interface FakeNSXPCConnection ()
@property id<SOSControlProtocol> control;
@end

NS_ASSUME_NONNULL_END
#endif /* FakeSOSControl_h */


