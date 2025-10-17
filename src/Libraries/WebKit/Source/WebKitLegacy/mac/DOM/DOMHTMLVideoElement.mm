/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#if ENABLE(VIDEO)

#import "DOMHTMLVideoElement.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLVideoElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLVideoElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLVideoElement

- (unsigned)width
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getUnsignedIntegralAttribute(WebCore::HTMLNames::widthAttr);
}

- (void)setWidth:(unsigned)newWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setUnsignedIntegralAttribute(WebCore::HTMLNames::widthAttr, newWidth);
}

- (unsigned)height
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getUnsignedIntegralAttribute(WebCore::HTMLNames::heightAttr);
}

- (void)setHeight:(unsigned)newHeight
{
    WebCore::JSMainThreadNullState state;
    IMPL->setUnsignedIntegralAttribute(WebCore::HTMLNames::heightAttr, newHeight);
}

- (unsigned)videoWidth
{
    WebCore::JSMainThreadNullState state;
    return IMPL->videoWidth();
}

- (unsigned)videoHeight
{
    WebCore::JSMainThreadNullState state;
    return IMPL->videoHeight();
}

- (NSString *)poster
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getURLAttribute(WebCore::HTMLNames::posterAttr).string();
}

- (void)setPoster:(NSString *)newPoster
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::posterAttr, newPoster);
}

- (BOOL)playsInline
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::playsinlineAttr);
}

- (void)setPlaysInline:(BOOL)newPlaysInline
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::playsinlineAttr, newPlaysInline);
}

- (BOOL)webkitSupportsFullscreen
{
    WebCore::JSMainThreadNullState state;
    return IMPL->webkitSupportsFullscreen();
}

- (BOOL)webkitDisplayingFullscreen
{
    WebCore::JSMainThreadNullState state;
    return IMPL->webkitDisplayingFullscreen();
}

- (void)webkitEnterFullscreen
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->webkitEnterFullscreen());
}

- (void)webkitExitFullscreen
{
    WebCore::JSMainThreadNullState state;
    IMPL->webkitExitFullscreen();
}

- (void)webkitEnterFullScreen
{
    [self webkitEnterFullscreen];
}

- (void)webkitExitFullScreen
{
    [self webkitExitFullscreen];
}

@end

#endif // ENABLE(VIDEO)

#undef IMPL
