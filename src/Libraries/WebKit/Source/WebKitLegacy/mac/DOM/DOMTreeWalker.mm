/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#import "DOMTreeWalkerInternal.h"

#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/NativeNodeFilter.h>
#import <WebCore/Node.h>
#import "ObjCNodeFilterCondition.h"
#import <WebCore/ThreadCheck.h>
#import <WebCore/TreeWalker.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#define IMPL reinterpret_cast<WebCore::TreeWalker*>(_internal)

@implementation DOMTreeWalker

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMTreeWalker class], self))
        return;

    if (_internal)
        IMPL->deref();
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

- (DOMNode *)currentNode
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->currentNode()));
}

- (void)setCurrentNode:(DOMNode *)newCurrentNode
{
    WebCore::JSMainThreadNullState state;
    ASSERT(newCurrentNode);

    if (!core(newCurrentNode))
        raiseTypeErrorException();
    IMPL->setCurrentNode(*core(newCurrentNode));
}

- (DOMNode *)parentNode
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->parentNode();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

- (DOMNode *)firstChild
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->firstChild();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

- (DOMNode *)lastChild
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->lastChild();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

- (DOMNode *)previousSibling
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->previousSibling();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

- (DOMNode *)nextSibling
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->nextSibling();
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

- (DOMNode *)nextNode
{
    WebCore::JSMainThreadNullState state;

    auto result = IMPL->nextNode();
    if (result.hasException())
        return nil;
    
    return kit(WTF::getPtr(result.releaseReturnValue()));
}

@end

DOMTreeWalker *kit(WebCore::TreeWalker* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMTreeWalker *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMTreeWalker alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
