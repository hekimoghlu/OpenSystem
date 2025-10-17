/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#import "DOMNamedNodeMapInternal.h"

#import "DOMNodeInternal.h"
#import "DOMInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/Attr.h>
#import <WebCore/JSExecState.h>
#import <WebCore/NamedNodeMap.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::NamedNodeMap*>(_internal)

@implementation DOMNamedNodeMap

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMNamedNodeMap class], self))
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

- (DOMNode *)getNamedItem:(NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->getNamedItem(name)));
}

- (DOMNode *)setNamedItem:(DOMNode *)node
{
    WebCore::JSMainThreadNullState state;
    if (!node)
        raiseTypeErrorException();
    auto& coreNode = *core(node);
    if (!is<WebCore::Attr>(coreNode))
        raiseTypeErrorException();
    return kit(raiseOnDOMError(IMPL->setNamedItem(downcast<WebCore::Attr>(coreNode))).get());
}

- (DOMNode *)removeNamedItem:(NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->removeNamedItem(name)).ptr());
}

- (DOMNode *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->item(index)));
}

- (DOMNode *)getNamedItemNS:(NSString *)namespaceURI localName:(NSString *)localName
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->getNamedItemNS(namespaceURI, localName)));
}

- (DOMNode *)setNamedItemNS:(DOMNode *)node
{
    return [self setNamedItem:node];
}

- (DOMNode *)removeNamedItemNS:(NSString *)namespaceURI localName:(NSString *)localName
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->removeNamedItemNS(namespaceURI, localName)).ptr());
}

@end

@implementation DOMNamedNodeMap (DOMNamedNodeMapDeprecated)

- (DOMNode *)getNamedItemNS:(NSString *)namespaceURI :(NSString *)localName
{
    return [self getNamedItemNS:namespaceURI localName:localName];
}

- (DOMNode *)removeNamedItemNS:(NSString *)namespaceURI :(NSString *)localName
{
    return [self removeNamedItemNS:namespaceURI localName:localName];
}

@end

DOMNamedNodeMap *kit(WebCore::NamedNodeMap* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMNamedNodeMap *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMNamedNodeMap alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
