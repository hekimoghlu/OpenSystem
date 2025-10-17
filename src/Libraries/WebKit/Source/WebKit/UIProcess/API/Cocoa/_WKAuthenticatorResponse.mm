/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#import "_WKAuthenticatorResponseInternal.h"

#import "_WKAuthenticationExtensionsClientOutputs.h"
#import <wtf/RetainPtr.h>

@implementation _WKAuthenticatorResponse {
    RetainPtr<_WKAuthenticationExtensionsClientOutputs> _extensions;
}

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensions:(RetainPtr<_WKAuthenticationExtensionsClientOutputs>&&)extensions attachment:(_WKAuthenticatorAttachment)attachment
{
    if (!(self = [super init]))
        return nil;

    _clientDataJSON = [clientDataJSON retain];
    _rawId = [rawId retain];
    _extensions = WTFMove(extensions);
    _attachment = attachment;

    return self;
}

- (instancetype)initWithClientDataJSON:(NSData *)clientDataJSON rawId:(NSData *)rawId extensionOutputsCBOR:(NSData *)extensionOutputsCBOR attachment:(_WKAuthenticatorAttachment)attachment
{
    if (!(self = [super init]))
        return nil;

    _clientDataJSON = [clientDataJSON retain];
    _rawId = [rawId retain];
    _extensionOutputsCBOR = [extensionOutputsCBOR copy];
    _attachment = attachment;

    return self;
}

- (void)dealloc
{
    [_clientDataJSON release];
    [_rawId release];
    [_extensionOutputsCBOR release];
    [super dealloc];
}

- (_WKAuthenticationExtensionsClientOutputs *)extensions
{
    return _extensions.get();
}

@end
