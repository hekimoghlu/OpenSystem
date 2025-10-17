/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#import "DOMMutationEvent.h"

#import "DOMEventInternal.h"
#import "DOMNodeInternal.h"
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/MutationEvent.h>
#import <WebCore/Node.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::MutationEvent*>(reinterpret_cast<WebCore::Event*>(_internal))

@implementation DOMMutationEvent

- (DOMNode *)relatedNode
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->relatedNode()));
}

- (NSString *)prevValue
{
    WebCore::JSMainThreadNullState state;
    return IMPL->prevValue();
}

- (NSString *)newValue
{
    WebCore::JSMainThreadNullState state;
    return IMPL->newValue();
}

- (NSString *)attrName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->attrName();
}

- (unsigned short)attrChange
{
    WebCore::JSMainThreadNullState state;
    return IMPL->attrChange();
}

- (void)initMutationEvent:(NSString *)type canBubble:(BOOL)canBubble cancelable:(BOOL)cancelable relatedNode:(DOMNode *)inRelatedNode prevValue:(NSString *)inPrevValue newValue:(NSString *)inNewValue attrName:(NSString *)inAttrName attrChange:(unsigned short)inAttrChange
{
    WebCore::JSMainThreadNullState state;
    IMPL->initMutationEvent(type, canBubble, cancelable, core(inRelatedNode), inPrevValue, inNewValue, inAttrName, inAttrChange);
}

@end

@implementation DOMMutationEvent (DOMMutationEventDeprecated)

- (void)initMutationEvent:(NSString *)type :(BOOL)canBubble :(BOOL)cancelable :(DOMNode *)inRelatedNode :(NSString *)inPrevValue :(NSString *)inNewValue :(NSString *)inAttrName :(unsigned short)inAttrChange
{
    [self initMutationEvent:type canBubble:canBubble cancelable:cancelable relatedNode:inRelatedNode prevValue:inPrevValue newValue:inNewValue attrName:inAttrName attrChange:inAttrChange];
}

@end

#undef IMPL
