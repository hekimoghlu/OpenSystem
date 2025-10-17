/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#import "DOMRangeInternal.h"

#import "DOMDocumentFragmentInternal.h"
#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/DocumentFragment.h>
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/Range.h>
#import <WebCore/SimpleRange.h>
#import <WebCore/TextIterator.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::Range*>(_internal)

@implementation DOMRange

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMRange class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (DOMNode *)startContainer
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->startContainer()));
}

- (int)startOffset
{
    WebCore::JSMainThreadNullState state;
    return IMPL->startOffset();
}

- (DOMNode *)endContainer
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->endContainer()));
}

- (int)endOffset
{
    WebCore::JSMainThreadNullState state;
    return IMPL->endOffset();
}

- (BOOL)collapsed
{
    WebCore::JSMainThreadNullState state;
    return IMPL->collapsed();
}

- (DOMNode *)commonAncestorContainer
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->commonAncestorContainer()));
}

- (NSString *)text
{
    WebCore::JSMainThreadNullState state;
    auto range = makeSimpleRange(*IMPL);
    range.start.document().updateLayout();
    return plainText(range);
}

- (void)setStart:(DOMNode *)refNode offset:(int)offset
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->setStart(*core(refNode), offset));
}

- (void)setEnd:(DOMNode *)refNode offset:(int)offset
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->setEnd(*core(refNode), offset));
}

- (void)setStartBefore:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->setStartBefore(*core(refNode)));
}

- (void)setStartAfter:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->setStartAfter(*core(refNode)));
}

- (void)setEndBefore:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->setEndBefore(*core(refNode)));
}

- (void)setEndAfter:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->setEndAfter(*core(refNode)));
}

- (void)collapse:(BOOL)toStart
{
    WebCore::JSMainThreadNullState state;
    IMPL->collapse(toStart);
}

- (void)selectNode:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->selectNode(*core(refNode)));
}

- (void)selectNodeContents:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->selectNodeContents(*core(refNode)));
}

- (short)compareBoundaryPoints:(unsigned short)how sourceRange:(DOMRange *)sourceRange
{
    WebCore::JSMainThreadNullState state;
    if (!sourceRange)
        raiseTypeErrorException();
    return raiseOnDOMError(IMPL->compareBoundaryPoints(how, *core(sourceRange)));
}

- (void)deleteContents
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->deleteContents());
}

- (DOMDocumentFragment *)extractContents
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->extractContents()).ptr());
}

- (DOMDocumentFragment *)cloneContents
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->cloneContents()).ptr());
}

- (void)insertNode:(DOMNode *)newNode
{
    WebCore::JSMainThreadNullState state;
    if (!newNode)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->insertNode(*core(newNode)));
}

- (void)surroundContents:(DOMNode *)newParent
{
    WebCore::JSMainThreadNullState state;
    if (!newParent)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->surroundContents(*core(newParent)));
}

- (DOMRange *)cloneRange
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->cloneRange()));
}

- (NSString *)toString
{
    WebCore::JSMainThreadNullState state;
    return IMPL->toString();
}

- (void)detach
{
    WebCore::JSMainThreadNullState state;
    IMPL->detach();
}

- (DOMDocumentFragment *)createContextualFragment:(NSString *)html
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->createContextualFragment(html)).ptr());
}

- (short)compareNode:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    return raiseOnDOMError(IMPL->compareNode(*core(refNode)));
}

- (BOOL)intersectsNode:(DOMNode *)refNode
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    return IMPL->intersectsNode(*core(refNode));
}

- (short)comparePoint:(DOMNode *)refNode offset:(int)offset
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    return raiseOnDOMError(IMPL->comparePoint(*core(refNode), offset));
}

- (BOOL)isPointInRange:(DOMNode *)refNode offset:(int)offset
{
    WebCore::JSMainThreadNullState state;
    if (!refNode)
        raiseTypeErrorException();
    return raiseOnDOMError(IMPL->isPointInRange(*core(refNode), offset));
}

- (void)expand:(NSString *)unit
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->expand(unit));
}

@end

@implementation DOMRange (DOMRangeDeprecated)

- (void)setStart:(DOMNode *)refNode :(int)offset
{
    [self setStart:refNode offset:offset];
}

- (void)setEnd:(DOMNode *)refNode :(int)offset
{
    [self setEnd:refNode offset:offset];
}

- (short)compareBoundaryPoints:(unsigned short)how :(DOMRange *)sourceRange
{
    return [self compareBoundaryPoints:how sourceRange:sourceRange];
}

@end

WebCore::Range* core(DOMRange *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::Range*>(wrapper->_internal) : 0;
}

DOMRange *kit(WebCore::Range* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMRange *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMRange alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

DOMRange *kit(const std::optional<WebCore::SimpleRange>& value)
{
    return kit(createLiveRange(value).get());
}

#undef IMPL
