/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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
#import "_WKPublicKeyCredentialCreationOptions.h"

#import "_WKPublicKeyCredentialRelyingPartyEntity.h"
#import "_WKPublicKeyCredentialUserEntity.h"

@implementation _WKPublicKeyCredentialCreationOptions

- (instancetype)initWithRelyingParty:(_WKPublicKeyCredentialRelyingPartyEntity *)relyingParty user:(_WKPublicKeyCredentialUserEntity *)user publicKeyCredentialParamaters:(NSArray<_WKPublicKeyCredentialParameters *> *)publicKeyCredentialParamaters
{
    if (!(self = [super init]))
        return nil;

    self.relyingParty = relyingParty;
    self.user = user;
    self.publicKeyCredentialParamaters = publicKeyCredentialParamaters;
    return self;
}

- (void)dealloc
{
    [_relyingParty release];
    [_user release];
    [_publicKeyCredentialParamaters release];
    [_timeout release];
    [_excludeCredentials release];
    [_authenticatorSelection release];
    [_extensions release];
    [_extensionsCBOR release];
    [super dealloc];
}

@end
