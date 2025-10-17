/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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
#import "WKWebProcessPlugInHitTestResultInternal.h"

#import "WKWebProcessPlugInNodeHandleInternal.h"
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKWebProcessPlugInHitTestResult {
    API::ObjectStorage<WebKit::InjectedBundleHitTestResult> _hitTestResult;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebProcessPlugInHitTestResult.class, self))
        return;
    _hitTestResult->~InjectedBundleHitTestResult();
    [super dealloc];
}

- (WKWebProcessPlugInNodeHandle *)nodeHandle
{
    return WebKit::wrapper(_hitTestResult->nodeHandle()).autorelease();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_hitTestResult;
}

@end
