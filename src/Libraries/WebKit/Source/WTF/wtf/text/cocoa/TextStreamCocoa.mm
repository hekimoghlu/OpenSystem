/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
#import <wtf/text/TextStreamCocoa.h>

#import <objc/runtime.h>
#import <wtf/text/cf/StringConcatenateCF.h>

namespace WTF {

TextStream& TextStream::operator<<(id object)
{
    if (object_isClass(object)) {
        m_text.append(NSStringFromClass(object));
        return *this;
    }

    auto outputArray = [&](NSArray *array) {
        *this << "[";

        for (NSUInteger i = 0; i < array.count; ++i) {
            id item = array[i];
            *this << item;
            if (i < array.count - 1)
                *this << ", ";
        }

        *this << "]";
    };

    if ([object isKindOfClass:[NSArray class]]) {
        outputArray(object);
        return *this;
    }

    auto outputDictionary = [&](NSDictionary *dictionary) {
        *this << "{";
        bool needLeadingComma = false;

        [dictionary enumerateKeysAndObjectsUsingBlock:[&](id key, id value, BOOL *) {
            if (needLeadingComma)
                *this << ", ";
            needLeadingComma = true;

            *this << key;
            *this << ": ";
            *this << value;
        }];

        *this << "}";
    };

    if ([object isKindOfClass:[NSDictionary class]]) {
        outputDictionary(object);
        return *this;
    }

    if ([object conformsToProtocol:@protocol(NSObject)])
        m_text.append([object description]);
    else
        *this << "(id)";

    return *this;
}

TextStream& operator<<(TextStream& ts, CGRect rect)
{
    ts << "{{" << rect.origin.x << ", " << rect.origin.y << "}, {" << rect.size.width << ", " << rect.size.height << "}}";
    return ts;
}

TextStream& operator<<(TextStream& ts, CGSize size)
{
    ts << "{" << size.width << ", " << size.height << "}";
    return ts;
}

TextStream& operator<<(TextStream& ts, CGPoint point)
{
    ts << "{" << point.x << ", " << point.y << "}";
    return ts;
}

} // namespace WTF
