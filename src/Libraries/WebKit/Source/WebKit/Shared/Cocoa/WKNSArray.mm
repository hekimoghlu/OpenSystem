/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#import "WKNSArray.h"

#import <WebCore/WebCoreObjCExtras.h>

@implementation WKNSArray {
    API::ObjectStorage<API::Array> _array;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKNSArray.class, self))
        return;

    _array->~Array();

    [super dealloc];
}

#pragma mark NSArray primitive methods

- (NSUInteger)count
{
    return _array->size();
}

- (id)objectAtIndex:(NSUInteger)i
{
    RefPtr object = self._protectedArray->at(i);
    return object ? (id)object->wrapper() : [NSNull null];
}

- (RefPtr<API::Array>)_protectedArray
{
    return _array.get();
}

#pragma mark NSCopying protocol implementation

- (id)copyWithZone:(NSZone *)zone
{
    return [self retain];
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_array;
}

@end
