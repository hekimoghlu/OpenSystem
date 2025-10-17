/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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
#import "WebTextIterator.h"

#import "DOMNodeInternal.h"
#import "DOMRangeInternal.h"
#import <JavaScriptCore/InitializeThreading.h>
#import <WebCore/Range.h>
#import <WebCore/TextIterator.h>
#import <WebCore/WebCoreJITOperations.h>
#import <wtf/MainThread.h>
#import <wtf/RunLoop.h>
#import <wtf/Vector.h>

@interface WebTextIteratorPrivate : NSObject {
@public
    std::unique_ptr<WebCore::TextIterator> _textIterator;
    Vector<unichar> _upconvertedText;
}
@end

@implementation WebTextIteratorPrivate

+ (void)initialize
{
#if !PLATFORM(IOS_FAMILY)
    JSC::initialize();
    WTF::initializeMainThread();
    WebCore::populateJITOperations();
#endif
}

@end

@implementation WebTextIterator

- (void)dealloc
{
    [_private release];
    [super dealloc];
}

- (id)initWithRange:(DOMRange *)range
{
    self = [super init];
    if (!self)
        return self;
    
    _private = [[WebTextIteratorPrivate alloc] init];
    if (!range)
        return self;

    _private->_textIterator = makeUnique<WebCore::TextIterator>(makeSimpleRange(*core(range)));
    return self;
}

- (void)advance
{
    if (_private->_textIterator)
        _private->_textIterator->advance();
    _private->_upconvertedText.shrink(0);
}

- (BOOL)atEnd
{
    return _private->_textIterator && _private->_textIterator->atEnd();
}

- (DOMRange *)currentRange
{
    if (!_private->_textIterator)
        return nil;
    auto& textIterator = *_private->_textIterator;
    if (textIterator.atEnd())
        return nil;
    return kit(textIterator.range());
}

// FIXME: Consider deprecating this method and creating one that does not require copying 8-bit characters.
- (const unichar*)currentTextPointer
{
    if (!_private->_textIterator)
        return nullptr;
    StringView text = _private->_textIterator->text();
    unsigned length = text.length();
    if (!length)
        return nullptr;
    if (!text.is8Bit())
        return reinterpret_cast<const unichar*>(text.span16().data());
    if (_private->_upconvertedText.isEmpty()) {
        auto characters = text.span8();
        _private->_upconvertedText.appendRange(characters.begin(), characters.end());
    }
    ASSERT(_private->_upconvertedText.size() == text.length());
    return _private->_upconvertedText.data();
}

- (NSUInteger)currentTextLength
{
    return _private->_textIterator ? _private->_textIterator->text().length() : 0;
}

@end

@implementation WebTextIterator (WebTextIteratorDeprecated)

- (DOMNode *)currentNode
{
    return _private->_textIterator ? kit(_private->_textIterator->node()) : nil;
}

- (NSString *)currentText
{
    return _private->_textIterator ? _private->_textIterator->text().createNSString().autorelease() : @"";
}

@end
