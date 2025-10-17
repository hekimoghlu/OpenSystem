/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
// FIXME (116158267): This file can be removed and its implementation merged directly into
// CDMInstanceSessionFairPlayStreamingAVFObjC once we no logner need to support a configuration
// where the BuiltInCDMKeyGroupingStrategyEnabled preference is off.

#if HAVE(AVCONTENTKEYSESSION)

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class AVContentKeyRequest;

@protocol WebAVContentKeyGrouping <NSObject>

@property (readonly, nullable) NSData *contentProtectionSessionIdentifier;

- (BOOL)associateContentKeyRequest:(AVContentKeyRequest *)contentKeyRequest;
- (void)expire;
- (void)processContentKeyRequestWithIdentifier:(nullable id)identifier initializationData:(nullable NSData *)initializationData options:(nullable NSDictionary<NSString *, id> *)options;

@end

NS_ASSUME_NONNULL_END

#endif // HAVE(AVCONTENTKEYSESSION)
