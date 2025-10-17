/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#import "DOMHTMLDocumentInternal.h"

#import "DOMHTMLCollectionInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/HTMLCollection.h>
#import <WebCore/HTMLDocument.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLDocument*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLDocument

- (DOMHTMLCollection *)embeds
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->embeds()));
}

- (DOMHTMLCollection *)plugins
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->embeds()));
}

- (DOMHTMLCollection *)scripts
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->scripts()));
}

- (int)width
{
    return 0;
}

- (int)height
{
    return 0;
}

- (NSString *)dir
{
    WebCore::JSMainThreadNullState state;
    return IMPL->dir();
}

- (void)setDir:(NSString *)newDir
{
    WebCore::JSMainThreadNullState state;
    IMPL->setDir(newDir);
}

- (NSString *)designMode
{
    WebCore::JSMainThreadNullState state;
    return IMPL->designMode();
}

- (void)setDesignMode:(NSString *)newDesignMode
{
    WebCore::JSMainThreadNullState state;
    IMPL->setDesignMode(newDesignMode);
}

- (NSString *)compatMode
{
    WebCore::JSMainThreadNullState state;
    return IMPL->compatMode();
}

- (NSString *)bgColor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->bgColor();
}

- (void)setBgColor:(NSString *)newBgColor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBgColor(newBgColor);
}

- (NSString *)fgColor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->fgColor();
}

- (void)setFgColor:(NSString *)newFgColor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setFgColor(newFgColor);
}

- (NSString *)alinkColor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->alinkColor();
}

- (void)setAlinkColor:(NSString *)newAlinkColor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAlinkColor(newAlinkColor);
}

- (NSString *)linkColor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->linkColorForBindings();
}

- (void)setLinkColor:(NSString *)newLinkColor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setLinkColorForBindings(newLinkColor);
}

- (NSString *)vlinkColor
{
    WebCore::JSMainThreadNullState state;
    return IMPL->vlinkColor();
}

- (void)setVlinkColor:(NSString *)newVlinkColor
{
    WebCore::JSMainThreadNullState state;
    IMPL->setVlinkColor(newVlinkColor);
}

- (void)open
{
    WebCore::JSMainThreadNullState state;
    IMPL->open();
}

- (void)close
{
    WebCore::JSMainThreadNullState state;
    IMPL->close();
}

- (void)write:(NSString *)text
{
    WebCore::JSMainThreadNullState state;
    IMPL->write(nullptr, FixedVector<String> { String { text } });
}

- (void)writeln:(NSString *)text
{
    WebCore::JSMainThreadNullState state;
    IMPL->writeln(nullptr, FixedVector<String> { String { text } });
}

- (void)clear
{
    WebCore::JSMainThreadNullState state;
    IMPL->clear();
}

- (void)captureEvents
{
    WebCore::JSMainThreadNullState state;
    IMPL->captureEvents();
}

- (void)releaseEvents
{
    WebCore::JSMainThreadNullState state;
    IMPL->releaseEvents();
}

@end

WebCore::HTMLDocument* core(DOMHTMLDocument *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLDocument*>(wrapper->_internal) : 0;
}

DOMHTMLDocument *kit(WebCore::HTMLDocument* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLDocument*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
