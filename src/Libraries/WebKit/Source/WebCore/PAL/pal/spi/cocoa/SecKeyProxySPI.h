/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#if HAVE(SEC_KEY_PROXY)

#if USE(APPLE_INTERNAL_SDK)

#import <Security/SecKeyProxy.h>

#else

#import <Foundation/Foundation.h>
#import <Security/SecBase.h>
#import <Security/SecKey.h>

NS_ASSUME_NONNULL_BEGIN
@interface SecKeyProxy : NSObject {
@private
    id _key;
    NSData * _Nullable _certificate;
    NSXPCListener *_listener;
}
// Creates new proxy instance. Proxy holds reference to the target key or identity and allows remote access to that target key as long as the proxy instance is kept alive.
- (instancetype)initWithKey:(SecKeyRef)key;
- (instancetype)initWithIdentity:(SecIdentityRef)identity;
// Retrieve endpoint to this proxy instance. Endpoint can be transferred over NSXPCConnection and passed to +[createKeyFromEndpoint:error:] method.
@property (readonly, nonatomic) NSXPCListenerEndpoint *endpoint;
// Invalidates all connections to this proxy.
- (void)invalidate;
// Creates new SecKey/SecIdentity object which forwards all operations to the target SecKey identified by endpoint. Returned SecKeyRef can be used as long as target SecKeyProxy instance is kept alive.
+ (nullable SecKeyRef)createKeyFromEndpoint:(NSXPCListenerEndpoint *)endpoint error:(NSError **)error;
+ (nullable SecIdentityRef)createIdentityFromEndpoint:(NSXPCListenerEndpoint *)endpoint error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END

#endif // USE(APPLE_INTERNAL_SDK)

#endif // HAVE(SEC_KEY_PROXY)
