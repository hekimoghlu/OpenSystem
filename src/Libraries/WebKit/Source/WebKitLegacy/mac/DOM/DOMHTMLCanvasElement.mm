/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
#import "DOMHTMLCanvasElement.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/HTMLCanvasElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/StringAdaptors.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLCanvasElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLCanvasElement

- (int)width
{
    WebCore::JSMainThreadNullState state;
    return IMPL->width();
}

- (void)setWidth:(int)newWidth
{
    WebCore::JSMainThreadNullState state;
    IMPL->setWidth(newWidth);
}

- (int)height
{
    WebCore::JSMainThreadNullState state;
    return IMPL->height();
}

- (void)setHeight:(int)newHeight
{
    WebCore::JSMainThreadNullState state;
    IMPL->setHeight(newHeight);
}

- (NSString *)toDataURL:(NSString *)type
{
    WebCore::JSMainThreadNullState state;

    auto exceptionOrReturnValue = IMPL->toDataURL(type);
    if (exceptionOrReturnValue.hasException())
        raiseDOMErrorException(exceptionOrReturnValue.releaseException());

    return exceptionOrReturnValue.releaseReturnValue().string;
}

@end

#undef IMPL
