/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#import "DOMHTMLCollectionInternal.h"

#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "DOMNodeListInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/HTMLCollection.h>
#import <WebCore/JSExecState.h>
#import <WebCore/Node.h>
#import <WebCore/NodeList.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::HTMLCollection*>(_internal)

@implementation DOMHTMLCollection

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMHTMLCollection class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (unsigned)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (DOMNode *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->item(index)));
}

- (DOMNode *)namedItem:(NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->namedItem(name)));
}

- (DOMNodeList *)tags:(NSString *)name
{
    if (!name)
        return nullptr;

    WebCore::JSMainThreadNullState state;
    return kit(static_cast<WebCore::NodeList*>(WTF::getPtr(IMPL->ownerNode().getElementsByTagName(name))));
}

@end

DOMHTMLCollection *kit(WebCore::HTMLCollection* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMHTMLCollection *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    RetainPtr<DOMHTMLCollection> wrapper = adoptNS([[kitClass(value) alloc] _init]);
    if (!wrapper)
        return nil;
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
