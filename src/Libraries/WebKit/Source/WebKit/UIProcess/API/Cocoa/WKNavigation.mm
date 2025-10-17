/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#import "WKNavigationInternal.h"
#import "WKWebpagePreferencesInternal.h"

#import "APINavigation.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKNavigation {
    API::ObjectStorage<API::Navigation> _navigation;
}

WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS;

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKNavigation.class, self))
        return;

    _navigation->~Navigation();

    [super dealloc];
}

- (NSURLRequest *)_request
{
    return _navigation->originalRequest().nsURLRequest(WebCore::HTTPBodyUpdatePolicy::DoNotUpdateHTTPBody);
}

- (BOOL)_isUserInitiated
{
    return _navigation->wasUserInitiated();
}

#if PLATFORM(IOS_FAMILY)

- (WKContentMode)effectiveContentMode
{
    return WebKit::contentMode(_navigation->effectiveContentMode());
}

#endif // PLATFORM(IOS_FAMILY)

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_navigation;
}

@end
