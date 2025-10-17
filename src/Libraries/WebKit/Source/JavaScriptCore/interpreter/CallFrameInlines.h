/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

#include "CallFrame.h"
#include "JSCalleeInlines.h"
#include "RegisterInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

inline Register& CallFrame::r(VirtualRegister reg)
{
    if (reg.isConstant())
        return *reinterpret_cast<Register*>(&this->codeBlock()->constantRegister(reg));
    return this[reg.offset()];
}

inline Register& CallFrame::uncheckedR(VirtualRegister reg)
{
    ASSERT(!reg.isConstant());
    return this[reg.offset()];
}

inline JSValue CallFrame::guaranteedJSValueCallee() const
{
    ASSERT(!callee().isNativeCallee());
    return this[static_cast<int>(CallFrameSlot::callee)].jsValue();
}

inline JSObject* CallFrame::jsCallee() const
{
    ASSERT(!callee().isNativeCallee());
    return this[static_cast<int>(CallFrameSlot::callee)].object();
}

inline CodeBlock* CallFrame::codeBlock() const
{
    ASSERT(!callee().isNativeCallee());
    return this[static_cast<int>(CallFrameSlot::codeBlock)].Register::codeBlock();
}

inline SUPPRESS_ASAN CodeBlock* CallFrame::unsafeCodeBlock() const
{
    return this[static_cast<int>(CallFrameSlot::codeBlock)].Register::asanUnsafeCodeBlock();
}

inline JSGlobalObject* CallFrame::lexicalGlobalObject(VM& vm) const
{
    if (callee().isNativeCallee())
        return lexicalGlobalObjectFromNativeCallee(vm);
    return jsCallee()->globalObject();
}

inline JSCell* CallFrame::codeOwnerCell() const
{
    if (callee().isNativeCallee())
        return codeOwnerCellSlow();
    return codeBlock();
}

inline bool CallFrame::isPartiallyInitializedFrame() const
{
    if (callee().isNativeCallee())
        return false;
    return jsCallee() == jsCallee()->globalObject()->partiallyInitializedFrameCallee();
}

inline bool CallFrame::isNativeCalleeFrame() const
{
    return callee().isNativeCallee();
}

inline void CallFrame::setCallee(JSObject* callee)
{
    static_cast<Register*>(this)[static_cast<int>(CallFrameSlot::callee)] = callee;
}

inline void CallFrame::setCallee(NativeCallee* callee)
{
    reinterpret_cast<uint64_t*>(this)[static_cast<int>(CallFrameSlot::callee)] = CalleeBits::encodeNativeCallee(callee);
}

inline void CallFrame::setCodeBlock(CodeBlock* codeBlock)
{
    static_cast<Register*>(this)[static_cast<int>(CallFrameSlot::codeBlock)] = codeBlock;
}

inline void CallFrame::setScope(int scopeRegisterOffset, JSScope* scope)
{
    static_cast<Register*>(this)[scopeRegisterOffset] = scope;
}

inline JSScope* CallFrame::scope(int scopeRegisterOffset) const
{
    ASSERT(this[scopeRegisterOffset].Register::scope());
    return this[scopeRegisterOffset].Register::scope();
}

inline Register* CallFrame::topOfFrame()
{
    if (!codeBlock())
        return registers();
    return topOfFrameInternal();
}

SUPPRESS_ASAN ALWAYS_INLINE void CallFrame::setCallSiteIndex(CallSiteIndex callSiteIndex)
{
    this[static_cast<int>(CallFrameSlot::argumentCountIncludingThis)].tag() = callSiteIndex.bits();
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
