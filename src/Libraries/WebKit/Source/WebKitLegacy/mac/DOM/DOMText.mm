/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#import "DOMTextInternal.h"

#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/Text.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::Text*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMText

- (NSString *)wholeText
{
    WebCore::JSMainThreadNullState state;
    return IMPL->wholeText();
}

- (DOMText *)splitText:(unsigned)offset
{
    WebCore::JSMainThreadNullState state;
    return kit(raiseOnDOMError(IMPL->splitText(offset)).ptr());
}

- (DOMText *)replaceWholeText:(NSString *)content
{
    WebCore::JSMainThreadNullState state;
    RefPtr { IMPL }->replaceWholeText(content);
    return self;
}

@end

DOMText *kit(WebCore::Text* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMText*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
