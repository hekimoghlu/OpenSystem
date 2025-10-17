/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#import "DOMCharacterData.h"

#import "DOMElementInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/Element.h>
#import "ExceptionHandlers.h"
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <WebCore/CharacterData.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::CharacterData*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMCharacterData

- (NSString *)data
{
    WebCore::JSMainThreadNullState state;
    return IMPL->data();
}

- (void)setData:(NSString *)newData
{
    WebCore::JSMainThreadNullState state;
    IMPL->setData(newData);
}

- (unsigned)length
{
    WebCore::JSMainThreadNullState state;
    return IMPL->length();
}

- (DOMElement *)previousElementSibling
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->previousElementSibling()));
}

- (DOMElement *)nextElementSibling
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->nextElementSibling()));
}

- (NSString *)substringData:(unsigned)offset length:(unsigned)inLength
{
    WebCore::JSMainThreadNullState state;
    return raiseOnDOMError(IMPL->substringData(offset, inLength));
}

- (void)appendData:(NSString *)inData
{
    WebCore::JSMainThreadNullState state;
    IMPL->appendData(inData);
}

- (void)insertData:(unsigned)offset data:(NSString *)inData
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->insertData(offset, inData));
}

- (void)deleteData:(unsigned)offset length:(unsigned)inLength
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->deleteData(offset, inLength));
}

- (void)replaceData:(unsigned)offset length:(unsigned)inLength data:(NSString *)inData
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->replaceData(offset, inLength, inData));
}

- (void)remove
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->remove());
}

@end

@implementation DOMCharacterData (DOMCharacterDataDeprecated)

- (NSString *)substringData:(unsigned)offset :(unsigned)inLength
{
    return [self substringData:offset length:inLength];
}

- (void)insertData:(unsigned)offset :(NSString *)inData
{
    [self insertData:offset data:inData];
}

- (void)deleteData:(unsigned)offset :(unsigned)inLength
{
    [self deleteData:offset length:inLength];
}

- (void)replaceData:(unsigned)offset :(unsigned)inLength :(NSString *)inData
{
    [self replaceData:offset length:inLength data:inData];
}

@end

#undef IMPL
