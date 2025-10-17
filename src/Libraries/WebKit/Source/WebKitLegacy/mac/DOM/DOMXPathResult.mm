/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#import "DOMXPathResultInternal.h"

#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/Node.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/XPathResult.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::XPathResult*>(_internal)

@implementation DOMXPathResult

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMXPathResult class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (unsigned short)resultType
{
    WebCore::JSMainThreadNullState state;
    return IMPL->resultType();
}

- (double)numberValue
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->numberValue());
}

- (NSString *)stringValue
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->stringValue());
}

- (BOOL)booleanValue
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->booleanValue());
}

- (DOMNode *)singleNodeValue
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->singleNodeValue()));
}

- (BOOL)invalidIteratorState
{
    WebCore::JSMainThreadNullState state;
    return IMPL->invalidIteratorState();
}

- (unsigned)snapshotLength
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->snapshotLength());
}

- (DOMNode *)iterateNext
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->iterateNext()));
}

- (DOMNode *)snapshotItem:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->snapshotItem(index)));
}

@end

WebCore::XPathResult* core(DOMXPathResult *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::XPathResult*>(wrapper->_internal) : nullptr;
}

DOMXPathResult *kit(WebCore::XPathResult* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMXPathResult *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMXPathResult alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
