/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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
#pragma once

#include "CodeBlock.h"
#include "JSScope.h"
#include "Register.h"

namespace JSC {

ALWAYS_INLINE CallFrame* Register::callFrame() const
{
    return u.callFrame;
}

ALWAYS_INLINE CodeBlock* Register::codeBlock() const
{
    return u.codeBlock;
}

SUPPRESS_ASAN ALWAYS_INLINE CodeBlock* Register::asanUnsafeCodeBlock() const
{
    return u.codeBlock;
}

ALWAYS_INLINE JSObject* Register::object() const
{
    return asObject(jsValue());
}

ALWAYS_INLINE Register& Register::operator=(CallFrame* callFrame)
{
    u.callFrame = callFrame;
    return *this;
}

ALWAYS_INLINE Register& Register::operator=(CodeBlock* codeBlock)
{
    u.codeBlock = codeBlock;
    return *this;
}

ALWAYS_INLINE Register& Register::operator=(JSCell* object)
{
    u.value = JSValue::encode(JSValue(object));
    return *this;
}

ALWAYS_INLINE Register& Register::operator=(JSScope* scope)
{
    *this = JSValue(scope);
    return *this;
}

ALWAYS_INLINE Register& Register::operator=(EncodedJSValue encodedJSValue)
{
    u.value = encodedJSValue;
    return *this;
}

ALWAYS_INLINE JSScope* Register::scope() const
{
    return jsCast<JSScope*>(unboxedCell());
}

} // namespace JSC
