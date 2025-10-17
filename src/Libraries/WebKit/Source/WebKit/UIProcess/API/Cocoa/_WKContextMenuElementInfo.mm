/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
#import "_WKContextMenuElementInfo.h"

#import "APIHitTestResult.h"
#import "APIString.h"
#import "_WKContextMenuElementInfoInternal.h"
#import "_WKHitTestResultInternal.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/RetainPtr.h>

#if !PLATFORM(IOS_FAMILY)

@implementation _WKContextMenuElementInfo

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKContextMenuElementInfo.class, self))
        return;
    _contextMenuElementInfoMac->API::ContextMenuElementInfoMac::~ContextMenuElementInfoMac();
    [super dealloc];
}

- (_WKHitTestResult *)hitTestResult
{
    auto& hitTestResultData = _contextMenuElementInfoMac->hitTestResultData();
    auto apiHitTestResult = API::HitTestResult::create(hitTestResultData, _contextMenuElementInfoMac->page());
    return retainPtr(wrapper(apiHitTestResult)).autorelease();
}

- (NSString *)qrCodePayloadString
{
    auto& qrCodePayloadString = _contextMenuElementInfoMac->qrCodePayloadString();
    return nsStringNilIfEmpty(qrCodePayloadString);
}

- (BOOL)hasEntireImage
{
    return _contextMenuElementInfoMac->hasEntireImage();
}

// MARK: WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_contextMenuElementInfoMac;
}

@end

#endif // !PLATFORM(IOS_FAMILY)
