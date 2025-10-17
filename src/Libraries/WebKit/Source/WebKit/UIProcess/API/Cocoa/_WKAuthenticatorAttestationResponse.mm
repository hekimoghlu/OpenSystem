/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#import "config.h"
#import "_WKAuthenticatorAttestationResponse.h"

#import "_WKAuthenticationExtensionsClientOutputs.h"
#import "_WKAuthenticatorResponseInternal.h"
#import <wtf/RetainPtr.h>

@implementation _WKAuthenticatorAttestationResponse

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensions:(RetainPtr<_WKAuthenticationExtensionsClientOutputs>&&)extensions attestationObject:(NSData *)attestationObject attachment:(_WKAuthenticatorAttachment)attachment transports:(NSArray<NSNumber *> *)transports
{
    if (!(self = [super initWithClientDataJSON:clientDataJSON rawId:rawId extensions:WTFMove(extensions) attachment:attachment]))
        return nil;

    _attestationObject = [attestationObject retain];
    _transports = [transports copy];
    return self;
}

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensionOutputsCBOR:(NSData *)extensionOutputsCBOR attestationObject:(NSData *)attestationObject attachment:(_WKAuthenticatorAttachment)attachment transports:(NSArray<NSNumber *> *)transports
{
    if (!(self = [super initWithClientDataJSON:clientDataJSON rawId:rawId extensionOutputsCBOR:WTFMove(extensionOutputsCBOR) attachment:attachment]))
        return nil;

    _attestationObject = [attestationObject retain];
    _transports = [transports copy];
    return self;
}

- (void)dealloc
{
    [_attestationObject release];
    [_transports release];
    [super dealloc];
}

@end
