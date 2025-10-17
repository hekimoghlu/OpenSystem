/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#include "config.h"
#include "StackFrame.h"

#include "CodeBlock.h"
#include "DebuggerPrimitives.h"
#include "JSCellInlines.h"
#include <wtf/text/MakeString.h>

namespace JSC {

StackFrame::StackFrame(VM& vm, JSCell* owner, JSCell* callee)
    : m_callee(vm, owner, callee)
{
}

StackFrame::StackFrame(VM& vm, JSCell* owner, JSCell* callee, CodeBlock* codeBlock, BytecodeIndex bytecodeIndex)
    : m_callee(vm, owner, callee)
    , m_codeBlock(vm, owner, codeBlock)
    , m_bytecodeIndex(bytecodeIndex)
{
}

StackFrame::StackFrame(VM& vm, JSCell* owner, CodeBlock* codeBlock, BytecodeIndex bytecodeIndex)
    : m_codeBlock(vm, owner, codeBlock)
    , m_bytecodeIndex(bytecodeIndex)
{
}

StackFrame::StackFrame(Wasm::IndexOrName indexOrName)
    : m_wasmFunctionIndexOrName(indexOrName)
    , m_isWasmFrame(true)
{
}

SourceID StackFrame::sourceID() const
{
    if (!m_codeBlock)
        return noSourceID;
    return m_codeBlock->ownerExecutable()->sourceID();
}

static String processSourceURL(VM& vm, const JSC::StackFrame& frame, const String& sourceURL)
{
    if (vm.clientData && (!protocolIsInHTTPFamily(sourceURL) && !protocolIs(sourceURL, "blob"_s))) {
        String overrideURL = vm.clientData->overrideSourceURL(frame, sourceURL);
        if (!overrideURL.isNull())
            return overrideURL;
    }

    if (!sourceURL.isNull())
        return sourceURL;
    return emptyString();
}

String StackFrame::sourceURL(VM& vm) const
{
    if (m_isWasmFrame)
        return "[wasm code]"_s;

    if (!m_codeBlock)
        return "[native code]"_s;

    return processSourceURL(vm, *this, m_codeBlock->ownerExecutable()->sourceURL());
}

String StackFrame::sourceURLStripped(VM& vm) const
{
    if (m_isWasmFrame)
        return "[wasm code]"_s;

    if (!m_codeBlock)
        return "[native code]"_s;

    return processSourceURL(vm, *this, m_codeBlock->ownerExecutable()->sourceURLStripped());
}

String StackFrame::functionName(VM& vm) const
{
    if (m_isWasmFrame)
        return makeString(m_wasmFunctionIndexOrName);

    if (m_codeBlock) {
        switch (m_codeBlock->codeType()) {
        case EvalCode:
            return "eval code"_s;
        case ModuleCode:
            return "module code"_s;
        case FunctionCode:
            break;
        case GlobalCode:
            return "global code"_s;
        default:
            ASSERT_NOT_REACHED();
        }
    }

    String name;
    if (m_callee) {
        if (m_callee->isObject())
            name = getCalculatedDisplayName(vm, jsCast<JSObject*>(m_callee.get())).impl();

        return name.isNull() ? emptyString() : name;
    }

    if (m_codeBlock) {
        if (auto* executable = jsDynamicCast<FunctionExecutable*>(m_codeBlock->ownerExecutable()))
            name = executable->ecmaName().impl();
    }

    return name.isNull() ? emptyString() : name;
}

LineColumn StackFrame::computeLineAndColumn() const
{
    if (!m_codeBlock)
        return { };

    auto lineColumn = m_codeBlock->lineColumnForBytecodeIndex(m_bytecodeIndex);

    ScriptExecutable* executable = m_codeBlock->ownerExecutable();
    if (std::optional<int> overrideLineNumber = executable->overrideLineNumber(m_codeBlock->vm()))
        lineColumn.line = overrideLineNumber.value();

    return lineColumn;
}

String StackFrame::toString(VM& vm) const
{
    String functionName = this->functionName(vm);
    String sourceURL = this->sourceURLStripped(vm);

    if (sourceURL.isEmpty() || !hasLineAndColumnInfo())
        return makeString(functionName, '@', sourceURL);

    auto lineColumn = computeLineAndColumn();
    return makeString(functionName, '@', sourceURL, ':', lineColumn.line, ':', lineColumn.column);
}

} // namespace JSC
