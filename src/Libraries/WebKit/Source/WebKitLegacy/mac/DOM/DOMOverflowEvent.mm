/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#import "DOMOverflowEvent.h"

#import "DOMEventInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/OverflowEvent.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>

#define IMPL static_cast<WebCore::OverflowEvent*>(reinterpret_cast<WebCore::Event*>(_internal))

@implementation DOMOverflowEvent

- (unsigned short)orient
{
    WebCore::JSMainThreadNullState state;
    return IMPL->orient();
}

- (BOOL)horizontalOverflow
{
    WebCore::JSMainThreadNullState state;
    return IMPL->horizontalOverflow();
}

- (BOOL)verticalOverflow
{
    WebCore::JSMainThreadNullState state;
    return IMPL->verticalOverflow();
}

- (void)initOverflowEvent:(unsigned short)inOrient horizontalOverflow:(BOOL)inHorizontalOverflow verticalOverflow:(BOOL)inVerticalOverflow
{
    WebCore::JSMainThreadNullState state;
    IMPL->initOverflowEvent(inOrient, inHorizontalOverflow, inVerticalOverflow);
}

@end

#undef IMPL
