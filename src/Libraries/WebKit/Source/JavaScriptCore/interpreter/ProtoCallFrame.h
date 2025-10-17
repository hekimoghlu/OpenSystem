/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#include "Register.h"
#include "StackAlignment.h"
#include <wtf/ForbidHeapAllocation.h>

#if ENABLE(WEBASSEMBLY)
#include "JSWebAssemblyInstance.h"
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

struct JS_EXPORT_PRIVATE ProtoCallFrame {
    WTF_FORBID_HEAP_ALLOCATION;
public:
    // CodeBlock, Callee, ArgumentCount, and |this|.
    static constexpr unsigned numberOfRegisters { 4 };

    Register codeBlockValue;
    Register calleeValue;
    Register argCountAndCodeOriginValue;
    Register thisArg;
    uint32_t paddedArgCount;
    EncodedJSValue* args;
    JSGlobalObject* globalObject;

    inline void init(CodeBlock*, JSGlobalObject*, JSObject*, JSValue, int, EncodedJSValue* otherArgs = nullptr);

    inline CodeBlock* codeBlock() const;
    inline void setCodeBlock(CodeBlock*);

    inline JSObject* callee() const;
    inline void setCallee(JSObject*);
    void setGlobalObject(JSGlobalObject* object)
    {
        globalObject = object;
    }

    int argumentCountIncludingThis() const { return argCountAndCodeOriginValue.payload(); }
    int argumentCount() const { return argumentCountIncludingThis() - 1; }
    void setArgumentCountIncludingThis(int count) { argCountAndCodeOriginValue.payload() = count; }
    void setPaddedArgCount(uint32_t argCount) { paddedArgCount = argCount; }

    void clearCurrentVPC() { argCountAndCodeOriginValue.tag() = 0; }
    
    JSValue thisValue() const { return thisArg.Register::jsValue(); }
    void setThisValue(JSValue value) { thisArg = value; }

#if ENABLE(WEBASSEMBLY)
    void setWasmInstance(JSWebAssemblyInstance* instance)
    {
        codeBlockValue = instance;
    }
#endif

    JSValue argument(size_t argumentIndex)
    {
        ASSERT(static_cast<int>(argumentIndex) < argumentCount());
        return JSValue::decode(args[argumentIndex]);
    }
    void setArgument(size_t argumentIndex, JSValue value)
    {
        ASSERT(static_cast<int>(argumentIndex) < argumentCount());
        args[argumentIndex] = JSValue::encode(value);
    }
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
