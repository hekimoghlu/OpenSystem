/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#import "DOMHTMLIFrameElement.h"

#import "DOMAbstractViewInternal.h"
#import "DOMDocumentInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/Document.h>
#import <WebCore/HTMLIFrameElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/JSExecState.h>
#import <WebCore/LocalDOMWindow.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLIFrameElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLIFrameElement

- (NSString *)align
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::alignAttr);
}

- (void)setAlign:(NSString *)newAlign
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::alignAttr, newAlign);
}

- (NSString *)frameBorder
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::frameborderAttr);
}

- (void)setFrameBorder:(NSString *)newFrameBorder
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::frameborderAttr, newFrameBorder);
}

- (NSString *)height
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::heightAttr);
}

- (void)setHeight:(NSString *)newHeight
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::heightAttr, newHeight);
}

- (NSString *)longDesc
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::longdescAttr);
}

- (void)setLongDesc:(NSString *)newLongDesc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::longdescAttr, newLongDesc);
}

- (NSString *)marginHeight
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::marginheightAttr);
}

- (void)setMarginHeight:(NSString *)newMarginHeight
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::marginheightAttr, newMarginHeight);
}

- (NSString *)marginWidth
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::marginwidthAttr);
}

- (void)setMarginWidth:(NSString *)newMarginWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::marginwidthAttr, newMarginWidth);
}

- (NSString *)name
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getNameAttribute();
}

- (void)setName:(NSString *)newName
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::nameAttr, newName);
}

- (NSString *)sandbox
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::sandboxAttr);
}

- (void)setSandbox:(NSString *)newSandbox
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::sandboxAttr, newSandbox);
}

- (NSString *)scrolling
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::scrollingAttr);
}

- (void)setScrolling:(NSString *)newScrolling
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::scrollingAttr, newScrolling);
}

- (NSString *)src
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::srcAttr).string();
}

- (void)setSrc:(NSString *)newSrc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::srcAttr, newSrc);
}

- (NSString *)srcdoc
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::srcdocAttr);
}

- (void)setSrcdoc:(NSString *)newSrcdoc
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::srcdocAttr, newSrcdoc);
}

- (NSString *)width
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::widthAttr);
}

- (void)setWidth:(NSString *)newWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::widthAttr, newWidth);
}

- (DOMDocument *)contentDocument
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->contentDocument()));
}

- (DOMAbstractView *)contentWindow
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->contentWindow()));
}

@end

#undef IMPL
