/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#import "_WKAuthenticatorAssertionResponse.h"

#import "_WKAuthenticationExtensionsClientOutputs.h"
#import "_WKAuthenticatorResponseInternal.h"
#import <wtf/RetainPtr.h>

@implementation _WKAuthenticatorAssertionResponse

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensions:(RetainPtr<_WKAuthenticationExtensionsClientOutputs>&&)extensions authenticatorData:(NSData *)authenticatorData signature:(NSData *)signature userHandle:(NSData *)userHandle attachment:(_WKAuthenticatorAttachment)attachment
{
    if (!(self = [super initWithClientDataJSON:clientDataJSON rawId:rawId extensions:WTFMove(extensions) attachment:attachment]))
        return nil;

    _authenticatorData = [authenticatorData retain];
    _signature = [signature retain];
    _userHandle = [userHandle retain];
    return self;
}

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensionOutputsCBOR:(NSData *)extensionOutputsCBOR authenticatorData:(NSData *)authenticatorData signature:(NSData *)signature userHandle:(NSData *)userHandle attachment:(_WKAuthenticatorAttachment)attachment
{
    if (!(self = [super initWithClientDataJSON:clientDataJSON rawId:rawId extensionOutputsCBOR:extensionOutputsCBOR attachment:attachment]))
        return nil;

    _authenticatorData = [authenticatorData retain];
    _signature = [signature retain];
    _userHandle = [userHandle retain];
    return self;
}

- (void)dealloc
{
    [_authenticatorData release];
    [_signature release];
    [_userHandle release];
    [super dealloc];
}

@end
