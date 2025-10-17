/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
#import "DOMXPathExpressionInternal.h"

#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "DOMXPathResultInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/Node.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/XPathExpression.h>
#import <WebCore/XPathResult.h>
#import <wtf/GetPtr.h>

#define IMPL reinterpret_cast<WebCore::XPathExpression*>(_internal)

@implementation DOMXPathExpression

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMXPathExpression class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (DOMXPathResult *)evaluate:(DOMNode *)contextNode type:(unsigned short)type inResult:(DOMXPathResult *)inResult
{
    if (!contextNode)
        return nullptr;

    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->evaluate(*core(contextNode), type, core(inResult))).ptr());
}

@end

@implementation DOMXPathExpression (DOMXPathExpressionDeprecated)

- (DOMXPathResult *)evaluate:(DOMNode *)contextNode :(unsigned short)type :(DOMXPathResult *)inResult
{
    return [self evaluate:contextNode type:type inResult:inResult];
}

@end

DOMXPathExpression *kit(WebCore::XPathExpression* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMXPathExpression *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMXPathExpression alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
