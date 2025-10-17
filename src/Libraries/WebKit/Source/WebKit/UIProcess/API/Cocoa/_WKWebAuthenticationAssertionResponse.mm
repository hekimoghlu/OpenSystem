/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#import "_WKWebAuthenticationAssertionResponseInternal.h"

#import "WKNSData.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKWebAuthenticationAssertionResponse

#if ENABLE(WEB_AUTHN)

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKWebAuthenticationAssertionResponse.class, self))
        return;

    _response->~WebAuthenticationAssertionResponse();

    [super dealloc];
}

- (NSString *)name
{
    return _response->name();
}

- (NSString *)displayName
{
    return _response->displayName();
}

- (NSData *)userHandle
{
    return wrapper(_response->userHandle()).autorelease();
}

- (BOOL)synchronizable
{
    return _response->synchronizable();
}

- (NSString *)group
{
    return _response->group();
}

- (NSData *)credentialID
{
    return wrapper(_response->credentialID()).autorelease();
}

- (NSString *)accessGroup
{
    return _response->accessGroup();
}

#endif // ENABLE(WEB_AUTHN)

- (void)setLAContext:(LAContext *)context
{
#if ENABLE(WEB_AUTHN)
    _response->setLAContext(context);
#endif
}

#if ENABLE(WEB_AUTHN)
#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_response;
}
#endif

@end
