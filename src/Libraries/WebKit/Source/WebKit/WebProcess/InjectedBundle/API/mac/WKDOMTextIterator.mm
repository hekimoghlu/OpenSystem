/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#import "WKDOMTextIterator.h"

#import "WKDOMInternals.h"
#import <WebCore/TextIterator.h>
#import <WebKit/WKDOMRange.h>

@interface WKDOMTextIterator () {
@private
    std::unique_ptr<WebCore::TextIterator> _textIterator;
    Vector<unichar> _upconvertedText;
}
@end

@implementation WKDOMTextIterator

- (id)initWithRange:(WKDOMRange *)range
{
    self = [super init];
    if (!self)
        return nil;

    if (!range)
        return self;

    _textIterator = makeUnique<WebCore::TextIterator>(makeSimpleRange(*WebKit::toWebCoreRange(range)));
    return self;
}

- (void)advance
{
    if (_textIterator)
        _textIterator->advance();
    _upconvertedText.shrink(0);
}

- (BOOL)atEnd
{
    return _textIterator && _textIterator->atEnd();
}

- (WKDOMRange *)currentRange
{
    return _textIterator ? WebKit::toWKDOMRange(createLiveRange(_textIterator->range()).ptr()) : nil;
}

// FIXME: Consider deprecating this method and creating one that does not require copying 8-bit characters.
- (const unichar*)currentTextPointer
{
    if (!_textIterator)
        return nullptr;
    StringView text = _textIterator->text();
    unsigned length = text.length();
    if (!length)
        return nullptr;
    if (!text.is8Bit())
        return reinterpret_cast<const unichar*>(text.span16().data());
    if (_upconvertedText.isEmpty()) {
        auto characters = text.span8();
        _upconvertedText.appendRange(characters.begin(), characters.end());
    }
    ASSERT(_upconvertedText.size() == text.length());
    return _upconvertedText.data();
}

- (NSUInteger)currentTextLength
{
    return _textIterator ? _textIterator->text().length() : 0;
}

@end
