/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#import <Foundation/Foundation.h>
#import "OTPairingPacketContext.h"
#import "OTPairingService.h"

#import "keychain/ot/OTDeviceInformationAdapter.h"

NS_ASSUME_NONNULL_BEGIN

@interface OTPairingSession : NSObject

@property (readonly) NSString *identifier;
@property (readwrite, nullable) OTPairingPacketContext *packet;
@property (readonly) KCPairingChannel *channel;
@property (readwrite) NSString *sentMessageIdentifier;

- (instancetype)initAsInitiator:(bool)initiator deviceInfo:(OTDeviceInformationActualAdapter *)deviceInfo identifier:(nullable NSString *)identifier;
- (instancetype)init NS_UNAVAILABLE;

#if !TARGET_OS_SIMULATOR
- (BOOL)acquireLockAssertion;
#endif /* !TARGET_OS_SIMULATOR */

- (void)addCompletionHandler:(OTPairingCompletionHandler)completionHandler;

- (void)didCompleteWithSuccess:(bool)success error:(NSError *)error;

@end

NS_ASSUME_NONNULL_END
