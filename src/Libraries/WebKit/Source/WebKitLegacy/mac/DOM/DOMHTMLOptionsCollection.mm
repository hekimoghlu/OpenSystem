/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#import "DOMHTMLOptionsCollectionInternal.h"

#import "DOMHTMLOptionElementInternal.h"
#import "DOMInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/HTMLOptGroupElement.h>
#import <WebCore/HTMLOptionElement.h>
#import <WebCore/HTMLOptionsCollection.h>
#import <WebCore/JSExecState.h>
#import <WebCore/Node.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <variant>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::HTMLOptionsCollection*>(_internal)

@implementation DOMHTMLOptionsCollection

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMHTMLOptionsCollection class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (int)selectedIndex
{
    WebCore::JSMainThreadNullState state;
    return IMPL->selectedIndex();
}

- (void)setSelectedIndex:(int)newSelectedIndex
{
    WebCore::JSMainThreadNullState state;
    IMPL->setSelectedIndex(newSelectedIndex);
}

- (unsigned)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (void)setLength:(unsigned)newLength
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setLength(newLength));
}

- (DOMNode *)namedItem:(NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->namedItem(name)));
}

- (void)add:(DOMHTMLOptionElement *)option index:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    if (!option)
        raiseTypeErrorException();
    raiseOnDOMError(IMPL->add(core(option), std::optional<WebCore::HTMLOptionsCollection::HTMLElementOrInt> { static_cast<int>(index) }));
}

- (void)remove:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    IMPL->remove(index);
}

- (DOMNode *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->item(index)));
}

@end

DOMHTMLOptionsCollection *kit(WebCore::HTMLOptionsCollection* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMHTMLOptionsCollection *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMHTMLOptionsCollection alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
