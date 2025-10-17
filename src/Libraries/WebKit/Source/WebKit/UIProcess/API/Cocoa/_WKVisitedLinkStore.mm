/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#import "_WKVisitedLinkStoreInternal.h"

#import "VisitedLinkStore.h"
#import <WebCore/SharedStringHash.h>
#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKVisitedLinkStore

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<WebKit::VisitedLinkStore>(self);

    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKVisitedLinkStore.class, self))
        return;

    Ref { *_visitedLinkStore }->~VisitedLinkStore();

    [super dealloc];
}

- (void)addVisitedLinkWithURL:(NSURL *)URL
{
    auto linkHash = WebCore::computeSharedStringHash(URL.absoluteString);

    Ref { *_visitedLinkStore }->addVisitedLinkHash(linkHash);
}

- (void)addVisitedLinkWithString:(NSString *)string
{
    Ref { *_visitedLinkStore }->addVisitedLinkHash(WebCore::computeSharedStringHash(string));
}

- (void)removeAll
{
    Ref { *_visitedLinkStore }->removeAll();
}

- (BOOL)containsVisitedLinkWithURL:(NSURL *)URL
{
    auto linkHash = WebCore::computeSharedStringHash(URL.absoluteString);

    return Ref { *_visitedLinkStore }->containsVisitedLinkHash(linkHash);
}

- (void)removeVisitedLinkWithURL:(NSURL *)URL
{
    auto linkHash = WebCore::computeSharedStringHash(URL.absoluteString);

    Ref { *_visitedLinkStore }->removeVisitedLinkHash(linkHash);
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_visitedLinkStore;
}

@end
