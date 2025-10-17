/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#ifndef CuttlefishXPCWrapper_h
#define CuttlefishXPCWrapper_h

#import <Foundation/Foundation.h>
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"

NS_ASSUME_NONNULL_BEGIN

// Wrapper around calling cuttlefish XPC calls that performs retries for generic
// XPC errors that warrant retrying. All operations are synchronous.

@interface CuttlefishXPCWrapper : NSObject<TrustedPeersHelperProtocol>
@property (readonly) id<NSXPCProxyCreating> cuttlefishXPCConnection;

- (instancetype) initWithCuttlefishXPCConnection: (id<NSXPCProxyCreating>)cuttlefishXPCConnection;

@end

NS_ASSUME_NONNULL_END

#endif // CuttlefishXPCWrapper_h
