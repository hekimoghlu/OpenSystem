/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#import "DOMInternal.h"

#import "DOMNodeIterator.h"

#import "DOMNodeInternal.h"
#import "DOMNodeIteratorInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/NativeNodeFilter.h>
#import <WebCore/Node.h>
#import <WebCore/NodeIterator.h>
#import "ObjCNodeFilterCondition.h"
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#define IMPL reinterpret_cast<WebCore::NodeIterator*>(_internal)

@implementation DOMNodeIterator

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMNodeIterator class], self))
        return;

    if (_internal) {
        [self detach];
        IMPL->deref();
    };
    [super dealloc];
}

- (DOMNode *)root
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->root()));
}

- (unsigned)whatToShow
{
    WebCore::JSMainThreadNullState state;
    return IMPL->whatToShow();
}

- (id <DOMNodeFilter>)filter
{
    WebCore::JSMainThreadNullState state;
    return kit(IMPL->filter());
}

- (BOOL)expandEntityReferences
{
    return NO;
}

- (DOMNode *)referenceNode
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->referenceNode()));
}

- (BOOL)pointerBeforeReferenceNode
{
    WebCore::JSMainThreadNullState state;
    return IMPL->pointerBeforeReferenceNode();
}

- (DOMNode *)nextNode
{
    WebCore::JSMainThreadNullState state;
    
    auto result = IMPL->nextNode();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

- (DOMNode *)previousNode
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->previousNode();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

- (void)detach
{
    WebCore::JSMainThreadNullState state;
    IMPL->detach();
}

@end

DOMNodeIterator *kit(WebCore::NodeIterator* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMNodeIterator *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMNodeIterator alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
