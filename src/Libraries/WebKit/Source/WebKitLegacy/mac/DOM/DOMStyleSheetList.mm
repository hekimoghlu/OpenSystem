/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

#import "DOMStyleSheetList.h"

#import "DOMNodeInternal.h"
#import "DOMStyleSheetInternal.h"
#import "DOMStyleSheetListInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/StyleSheet.h>
#import <WebCore/StyleSheetList.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#define IMPL reinterpret_cast<WebCore::StyleSheetList*>(_internal)

@implementation DOMStyleSheetList

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMStyleSheetList class], self))
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

- (DOMStyleSheet *)item:(unsigned)index
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->item(index)));
}

@end

DOMStyleSheetList *kit(WebCore::StyleSheetList* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMStyleSheetList *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMStyleSheetList alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
