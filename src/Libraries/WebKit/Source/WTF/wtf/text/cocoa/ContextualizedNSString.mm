/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#import <wtf/text/cocoa/ContextualizedNSString.h>

#import <algorithm>
#import <wtf/text/StringView.h>

@implementation WTFContextualizedNSString {
    StringView context;
    StringView contents;
}

- (instancetype)initWithContext:(StringView)context contents:(StringView)contents
{
    if (self = [super init]) {
        self->context = context;
        self->contents = contents;
    }
    return self;
}

- (NSUInteger)length
{
    return context.length() + contents.length();
}

- (unichar)characterAtIndex:(NSUInteger)index
{
    if (index < context.length())
        return context[index];
    return contents[index - context.length()];
}

- (void)getCharacters:(unichar *)buffer range:(NSRange)range
{
    auto contextLow = std::clamp(static_cast<unsigned>(range.location), 0U, context.length());
    auto contextHigh = std::clamp(static_cast<unsigned>(range.location + range.length), 0U, context.length());
    auto contextSubstring = context.substring(contextLow, contextHigh - contextLow);
    auto contentsLow = std::clamp(static_cast<unsigned>(range.location), context.length(), context.length() + contents.length());
    auto contentsHigh = std::clamp(static_cast<unsigned>(range.location + range.length), context.length(), context.length() + contents.length());
    auto contentsSubstring = contents.substring(contentsLow - context.length(), contentsHigh - contentsLow);
    static_assert(std::is_same_v<std::make_unsigned_t<unichar>, std::make_unsigned_t<UChar>>);
    // FIXME: We don't actually know the size of buffer here.
    auto bufferSpan = unsafeMakeSpan(reinterpret_cast<UChar*>(buffer), contextSubstring.length() + contentsSubstring.length());
    contextSubstring.getCharacters(bufferSpan);
    contentsSubstring.getCharacters(bufferSpan.subspan(contextSubstring.length()));
}

@end
