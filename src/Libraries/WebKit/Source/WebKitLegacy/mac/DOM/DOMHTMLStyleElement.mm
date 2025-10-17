/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#import "DOMHTMLStyleElementInternal.h"

#import "DOMNodeInternal.h"
#import "DOMStyleSheetInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/HTMLNames.h>
#import <WebCore/HTMLStyleElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/StyleSheet.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLStyleElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLStyleElement

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

- (NSString *)media
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::mediaAttr);
}

- (void)setMedia:(NSString *)newMedia
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::mediaAttr, newMedia);
}

- (NSString *)type
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::typeAttr);
}

- (void)setType:(NSString *)newType
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::typeAttr, newType);
}

- (DOMStyleSheet *)sheet
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->sheet()));
}

- (NSString *)nonce
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::nonceAttr);
}

- (void)setNonce:(NSString *)newNonce
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::nonceAttr, newNonce);
}

@end

WebCore::HTMLStyleElement* core(DOMHTMLStyleElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLStyleElement*>(wrapper->_internal) : 0;
}

#undef IMPL
