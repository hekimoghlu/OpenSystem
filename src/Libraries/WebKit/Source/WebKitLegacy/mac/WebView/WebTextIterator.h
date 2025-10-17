/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#import <Foundation/Foundation.h>

@class DOMRange;
@class DOMNode;
@class WebTextIteratorPrivate;

@interface WebTextIterator : NSObject {
@private
    WebTextIteratorPrivate *_private;
}

- (id)initWithRange:(DOMRange *)range;

/*!
 @method advance
 @abstract Moves the WebTextIterator to the next bit of text or boundary between runs of text.
 The iterator can break up runs of text however it finds convenient, so clients need to handle
 text runs that are broken up into arbitrary pieces.
 */
- (void)advance;

/*!
 @method atEnd
 @result YES if the WebTextIterator has reached the end of the DOMRange.
 */
- (BOOL)atEnd;

/*!
 @method currentTextLength
 @result Length of the current text. Length of zero means that the iterator is at a boundary,
 such as an image, that separates runs of text.
 */
- (NSUInteger)currentTextLength;

/*!
 @method currentTextPointer
 @result A pointer to the current text. Like the WebTextIterator itself, the pointer becomes
 invalid after any modification is made to the document; it must be used before the document
 is changed or the iterator is advanced.
 */
- (const unichar *)currentTextPointer;

/*!
 @method currentRange
 @abstract A function that identifies the specific document range that text corresponds to.
 This can be quite costly to compute for non-text items, so when possible this should only
 be called once the caller has determined that the text is text it wants to process. If you
 call currentRange every time you advance the iterator, performance will be extremely slow
 due to the cost of computing a DOM range.
 @result A DOM range indicating the position within the document of the current text.
 */
- (DOMRange *)currentRange;

@end

@interface WebTextIterator (WebTextIteratorDeprecated)

/*!
 @method currentNode
 @abstract A convenience method that finds the first node in currentRange; it's almost always better to use currentRange instead.
 @result The current DOMNode in the WebTextIterator
 */
- (DOMNode *)currentNode;

/*!
 @method currentText
 @abstract A convenience method that makes an NSString out of the current text; it's almost always better to use currentTextPointer and currentTextLength instead.
 @result The current text in the WebTextIterator.
 */
- (NSString *)currentText;

@end
