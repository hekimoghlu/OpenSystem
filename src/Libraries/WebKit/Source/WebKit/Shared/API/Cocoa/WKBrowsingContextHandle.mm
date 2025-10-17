/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 11, 2024.
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
#import "WKBrowsingContextHandleInternal.h"

#import "WebPage.h"
#import "WebPageProxy.h"
#import <wtf/HashFunctions.h>

@implementation WKBrowsingContextHandle

- (id)_initWithPageProxy:(NakedRef<WebKit::WebPageProxy>)page
{
    return [self _initWithPageProxyID:page->identifier() andWebPageID:page->webPageIDInMainFrameProcess()];
}

- (id)_initWithPage:(NakedRef<WebKit::WebPage>)page
{
    return [self _initWithPageProxyID:page->webPageProxyIdentifier() andWebPageID:page->identifier()];
}

- (id)_initWithPageProxyID:(WebKit::WebPageProxyIdentifier)pageProxyID andWebPageID:(WebCore::PageIdentifier)webPageID
{
    if (!(self = [super init]))
        return nil;

    _pageProxyID = pageProxyID;
    _webPageID = webPageID.toUInt64();

    return self;
}

- (NSUInteger)hash
{
    return computeHash(*_pageProxyID, _webPageID);
}

- (BOOL)isEqual:(id)object
{
    if (![object isKindOfClass:[WKBrowsingContextHandle class]])
        return NO;

    return _pageProxyID == static_cast<WKBrowsingContextHandle *>(object)->_pageProxyID && _webPageID == static_cast<WKBrowsingContextHandle *>(object)->_webPageID;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
    [coder encodeInt64:_pageProxyID->toUInt64() forKey:@"pageProxyID"];
    [coder encodeInt64:_webPageID forKey:@"webPageID"];
}

- (id)initWithCoder:(NSCoder *)coder
{
    return [self _initWithPageProxyID:ObjectIdentifier<WebKit::WebPageProxyIdentifierType>([coder decodeInt64ForKey:@"pageProxyID"]) andWebPageID:ObjectIdentifier<WebCore::PageIdentifierType>([coder decodeInt64ForKey:@"webPageID"])];
}

+ (BOOL)supportsSecureCoding
{
    return YES;
}

- (id)copyWithZone:(NSZone *)zone
{
    return [[WKBrowsingContextHandle allocWithZone:zone] _initWithPageProxyID:*_pageProxyID andWebPageID:ObjectIdentifier<WebCore::PageIdentifierType>(_webPageID)];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; pageProxyID = %llu; webPageID = %llu>", NSStringFromClass(self.class), self, _pageProxyID->toUInt64(), _webPageID];
}
@end
