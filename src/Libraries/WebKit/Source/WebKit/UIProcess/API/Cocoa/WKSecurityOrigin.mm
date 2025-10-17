/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#import "WKSecurityOriginInternal.h"

#import <WebCore/ResourceRequest.h>
#import <WebCore/SecurityOrigin.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/RefPtr.h>

@implementation WKSecurityOrigin

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKSecurityOrigin.class, self))
        return;

    _securityOrigin->~SecurityOrigin();

    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; protocol = %@; host = %@; port = %li>", NSStringFromClass(self.class), self, self.protocol, self.host, (long)self.port];
}

- (NSString *)protocol
{
    return _securityOrigin->securityOrigin().protocol();
}

- (NSString *)host
{
    return _securityOrigin->securityOrigin().host();
}

- (NSInteger)port
{
    return _securityOrigin->securityOrigin().port().value_or(0);
}

-(BOOL)isSameSiteAsOrigin:(WKSecurityOrigin *)origin
{
    auto thisOrigin = _securityOrigin->securityOrigin().securityOrigin();
    auto otherOrigin = origin->_securityOrigin->securityOrigin().securityOrigin();

    return thisOrigin->isSameSiteAs(otherOrigin.get());
}

-(BOOL)isSameSiteAsURL:(NSURL *)url
{
    auto thisOrigin = _securityOrigin->securityOrigin().securityOrigin();
    auto otherOrigin = WebCore::SecurityOrigin::create(URL { url });

    return thisOrigin->isSameSiteAs(otherOrigin.get());
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_securityOrigin;
}

@end
