/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#import "WKNSDictionary.h"

#import "WKNSArray.h"
#import <WebCore/WebCoreObjCExtras.h>

using namespace WebKit;

@implementation WKNSDictionary {
    API::ObjectStorage<API::Dictionary> _dictionary;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKNSDictionary.class, self))
        return;

    _dictionary->~Dictionary();

    [super dealloc];
}

#pragma mark NSDictionary primitive methods

- (instancetype)initWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)count
{
    ASSERT_NOT_REACHED();
    self = [super initWithObjects:objects forKeys:keys count:count];
    return self;
}

- (NSUInteger)count
{
    return _dictionary->size();
}

- (id)objectForKey:(id)key
{
    auto *str = dynamic_objc_cast<NSString>(key);
    if (!str)
        return nil;

    bool exists;
    RefPtr value = _dictionary->get(str, exists);
    if (!exists)
        return nil;

    return value ? (id)value->wrapper() : [NSNull null];
}

- (NSEnumerator *)keyEnumerator
{
    return [wrapper(_dictionary->keys()) objectEnumerator];
}

#pragma mark NSCopying protocol implementation

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_dictionary;
}

@end
