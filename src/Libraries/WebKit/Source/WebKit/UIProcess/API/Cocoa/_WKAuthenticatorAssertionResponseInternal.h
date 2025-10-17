/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#import "_WKAuthenticatorAssertionResponse.h"
#import <wtf/RetainPtr.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKAuthenticationExtensionsClientOutputs;

@interface _WKAuthenticatorAssertionResponse ()

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensions:(RetainPtr<_WKAuthenticationExtensionsClientOutputs>&&)extensions authenticatorData:(NSData *)authenticatorData signature:(NSData *)signature userHandle:(NSData * _Nullable)userHandle attachment:(_WKAuthenticatorAttachment)attachment;

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensionOutputsCBOR:(NSData *)extensionOutputsCBOR authenticatorData:(NSData *)authenticatorData signature:(NSData *)signature userHandle:(NSData * _Nullable)userHandle attachment:(_WKAuthenticatorAttachment)attachment;

@end

NS_ASSUME_NONNULL_END
