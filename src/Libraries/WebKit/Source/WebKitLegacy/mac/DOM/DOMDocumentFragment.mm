/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#import "DOMDocumentFragmentInternal.h"

#import "DOMElementInternal.h"
#import "DOMHTMLCollectionInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/DocumentFragment.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>

#define IMPL static_cast<WebCore::DocumentFragment*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMDocumentFragment

- (DOMHTMLCollection *)children
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->children()));
}

- (DOMElement *)firstElementChild
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->firstElementChild()));
}

- (DOMElement *)lastElementChild
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->lastElementChild()));
}

- (unsigned)childElementCount
{
    WebCore::JSMainThreadNullState state;
    return IMPL->childElementCount();
}

- (DOMElement *)getElementById:(NSString *)elementId
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->getElementById(AtomString(elementId))));
}

- (DOMElement *)querySelector:(NSString *)selectors
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->querySelector(selectors)));
}

- (DOMNodeList *)querySelectorAll:(NSString *)selectors
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->querySelectorAll(selectors)).ptr());
}

@end

WebCore::DocumentFragment* core(DOMDocumentFragment *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::DocumentFragment*>(wrapper->_internal) : 0;
}

DOMDocumentFragment *kit(WebCore::DocumentFragment* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMDocumentFragment*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
