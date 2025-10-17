/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#pragma once

#if HAVE(RSA_BSSA)

#if USE(APPLE_INTERNAL_SDK)

// FIXME(227598): Remove conditional once CryptoKitPrivate/RSABSSA.h is available.
#if __has_include(<CryptoKitPrivate/RSABSSA.h>)
#import <CryptoKitPrivate/RSABSSA.h>
#else
#import <CryptoKitCBridging/RSABSSA.h>
#endif

#else

@interface RSABSSATokenWaitingActivation : NSObject
#if HAVE(RSA_BSSA)
- (RSABSSATokenReady*)activateTokenWithServerResponse:(NSData*)serverResponse error:(NSError* __autoreleasing *)error;
#endif
@property (nonatomic, retain, readonly) NSData* blindedMessage;
@end

@interface RSABSSATokenReady : NSObject
@property (nonatomic, retain, readonly) NSData* tokenContent;
@property (nonatomic, retain, readonly) NSData* keyId;
@property (nonatomic, retain, readonly) NSData* signature;
@end

#if HAVE(RSA_BSSA)
@interface RSABSSATokenBlinder : NSObject
- (instancetype)initWithPublicKey:(NSData*)spkiBytes error:(NSError* __autoreleasing *)error;
- (RSABSSATokenWaitingActivation*)tokenWaitingActivationWithContent:(NSData*)content error:(NSError* __autoreleasing *)error;
@property (nonatomic, retain, readonly) NSData* keyId;
@end
#endif

#endif // USE(APPLE_INTERNAL_SDK)

#endif // HAVE(RSA_BSSA)
