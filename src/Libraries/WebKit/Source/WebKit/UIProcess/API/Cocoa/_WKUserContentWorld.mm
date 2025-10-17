/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#import "_WKUserContentWorldInternal.h"

#import "WKContentWorldInternal.h"

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
@implementation _WKUserContentWorld

- (instancetype)_initWithName:(NSString *)name
{
    if (!(self = [super init]))
        return nil;

    _contentWorld = [WKContentWorld worldWithName:name];
    return self;
}

- (instancetype)_init
{
    if (!(self = [super init]))
        return nil;

    _contentWorld = [WKContentWorld pageWorld];
    return self;
}

- (instancetype)_initWithContentWorld:(WKContentWorld *)world
{
    if (!(self = [super init]))
        return nil;

    _contentWorld = world;
    return self;
}

+ (_WKUserContentWorld *)worldWithName:(NSString *)name
{
    return adoptNS([[_WKUserContentWorld alloc] _initWithName:name]).autorelease();
}

+ (_WKUserContentWorld *)normalWorld
{
    return adoptNS([[_WKUserContentWorld alloc] _init]).autorelease();
}

- (NSString *)name
{
    return [_contentWorld name];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return [_contentWorld _apiObject];
}

@end
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
