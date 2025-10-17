/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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
#import "WKNavigationResponseInternal.h"

#import "WKFrameInfoInternal.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKNavigationResponse

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKNavigationResponse.class, self))
        return;

    _navigationResponse->~NavigationResponse();

    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; response = %@>", NSStringFromClass(self.class), self, self.response];
}

- (BOOL)isForMainFrame
{
    return _navigationResponse->frame().isMainFrame();
}

- (NSURLResponse *)response
{
    return _navigationResponse->response().nsURLResponse();
}

- (BOOL)canShowMIMEType
{
    return _navigationResponse->canShowMIMEType();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_navigationResponse;
}

@end

@implementation WKNavigationResponse (WKPrivate)

- (WKFrameInfo *)_frame
{
    // FIXME: This RefPtr should not be necessary. Remove it once clang static analyzer is fixed.
    return wrapper(RefPtr { _navigationResponse.get() }->protectedFrame().get());
}

- (WKFrameInfo *)_navigationInitiatingFrame
{
    return wrapper(_navigationResponse->navigationInitiatingFrame());
}

- (NSURLRequest *)_request
{
    return _navigationResponse->request().nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
}

- (NSString *)_downloadAttribute
{
    const String& attribute = _navigationResponse->downloadAttribute();
    return attribute.isNull() ? nil : (NSString *)attribute;
}

- (BOOL)_wasPrivateRelayed
{
    return _navigationResponse->response().wasPrivateRelayed();
}

- (NSString *)_proxyName
{
    return _navigationResponse->response().proxyName();
}

- (BOOL)_isFromNetwork
{
    return _navigationResponse->response().source() == WebCore::ResourceResponseBase::Source::Network;
}
@end
