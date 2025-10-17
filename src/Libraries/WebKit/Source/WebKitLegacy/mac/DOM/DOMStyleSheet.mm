/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 3, 2025.
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

#import "DOMStyleSheet.h"

#import "DOMMediaListInternal.h"
#import "DOMNodeInternal.h"
#import "DOMStyleSheetInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/MediaList.h>
#import <WebCore/Node.h>
#import <WebCore/StyleSheet.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL reinterpret_cast<WebCore::StyleSheet*>(_internal)

@implementation DOMStyleSheet

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainThread([DOMStyleSheet class], self))
        return;

    if (_internal)
        IMPL->deref();
    [super dealloc];
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->type();
}

- (BOOL)disabled
{
    WebCore::JSMainThreadNullState state;
    return IMPL->disabled();
}

- (void)setDisabled:(BOOL)newDisabled
{
    WebCore::JSMainThreadNullState state;
    IMPL->setDisabled(newDisabled);
}

- (DOMNode *)ownerNode
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->ownerNode()));
}

- (DOMStyleSheet *)parentStyleSheet
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->parentStyleSheet()));
}

- (NSString *)href
{
    WebCore::JSMainThreadNullState state;
    return IMPL->href();
}

- (NSString *)title
{
    WebCore::JSMainThreadNullState state;
    return IMPL->title();
}

- (DOMMediaList *)media
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->media()));
}

@end

DOMStyleSheet *kit(WebCore::StyleSheet* value)
{
    WebCoreThreadViolationCheckRoundOne();
    if (!value)
        return nil;
    if (DOMStyleSheet *wrapper = getDOMWrapper(value))
        return retainPtr(wrapper).autorelease();
    RetainPtr<DOMStyleSheet> wrapper = adoptNS([[kitClass(value) alloc] _init]);
    if (!wrapper)
        return nil;
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(value);
    value->ref();
    addDOMWrapper(wrapper.get(), value);
    return wrapper.autorelease();
}

#undef IMPL
