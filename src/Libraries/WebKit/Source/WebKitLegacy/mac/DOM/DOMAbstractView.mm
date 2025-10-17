/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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
#import "DOMAbstractViewInternal.h"

#import "DOMDocumentInternal.h"
#import "DOMInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/Document.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/WindowProxy.h>

#define IMPL reinterpret_cast<WebCore::LocalFrame*>(_internal)

@implementation DOMAbstractView

- (void)dealloc
{
    WebCoreThreadViolationCheckRoundOne();
    [super dealloc];
}

- (DOMDocument *)document
{
    if (!_internal)
        return nil;
    return kit(IMPL->document());
}

@end

@implementation DOMAbstractView (WebKitLegacyInternal)

- (void)_disconnectFrame
{
    ASSERT(_internal);
    removeDOMWrapper(_internal);
    _internal = 0;
}

@end

WebCore::LocalDOMWindow* core(DOMAbstractView *wrapper)
{
    if (!wrapper)
        return 0;
    if (!wrapper->_internal)
        return 0;
    return reinterpret_cast<WebCore::LocalFrame*>(wrapper->_internal)->document()->domWindow();
}

DOMAbstractView *kit(WebCore::LocalDOMWindow* value)
{
    WebCoreThreadViolationCheckRoundOne();

    if (!value)
        return nil;
    auto* frame = value->frame();
    if (!frame)
        return nil;
    if (DOMAbstractView *wrapper = getDOMWrapper(frame))
        return retainPtr(wrapper).autorelease();
    auto wrapper = adoptNS([[DOMAbstractView alloc] _init]);
    wrapper->_internal = reinterpret_cast<DOMObjectInternal*>(frame);
    addDOMWrapper(wrapper.get(), frame);
    return wrapper.autorelease();
}

DOMAbstractView *kit(WebCore::DOMWindow* value)
{
    if (!is<WebCore::LocalDOMWindow>(value))
        return nil;

    return kit(downcast<WebCore::LocalDOMWindow>(value));
}

DOMAbstractView *kit(WebCore::WindowProxy* windowProxy)
{
    if (!windowProxy)
        return nil;

    return kit(windowProxy->window());
}

WebCore::WindowProxy* toWindowProxy(DOMAbstractView *view)
{
    auto* window = core(view);
    if (!window || !window->frame())
        return nil;
    return &window->frame()->windowProxy();
}

#undef IMPL
