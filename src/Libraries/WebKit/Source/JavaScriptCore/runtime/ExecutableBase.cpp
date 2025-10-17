/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 8, 2023.
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

#include "EvalCodeBlock.h"
#include "FunctionCodeBlock.h"
#include "JSCellInlines.h"
#include "ModuleProgramCodeBlock.h"
#include "NativeExecutable.h"
#include "ProgramCodeBlock.h"
#include <wtf/CommaPrinter.h>

namespace JSC {

const ClassInfo ExecutableBase::s_info = { "Executable"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(ExecutableBase) };

void ExecutableBase::destroy(JSCell* cell)
{
    static_cast<ExecutableBase*>(cell)->ExecutableBase::~ExecutableBase();
}

void ExecutableBase::dump(PrintStream& out) const
{
    ExecutableBase* realThis = const_cast<ExecutableBase*>(this);

    switch (type()) {
    case NativeExecutableType: {
        NativeExecutable* native = jsCast<NativeExecutable*>(realThis);
        out.print("NativeExecutable:"_s, RawPointer(native->function().taggedPtr()), "/"_s, RawPointer(native->constructor().taggedPtr()));
        return;
    }
    case EvalExecutableType: {
        EvalExecutable* eval = jsCast<EvalExecutable*>(realThis);
        if (CodeBlock* codeBlock = eval->codeBlock())
            out.print(*codeBlock);
        else
            out.print("EvalExecutable w/o CodeBlock"_s);
        return;
    }
    case ProgramExecutableType: {
        ProgramExecutable* eval = jsCast<ProgramExecutable*>(realThis);
        if (CodeBlock* codeBlock = eval->codeBlock())
            out.print(*codeBlock);
        else
            out.print("ProgramExecutable w/o CodeBlock"_s);
        return;
    }
    case ModuleProgramExecutableType: {
        ModuleProgramExecutable* executable = jsCast<ModuleProgramExecutable*>(realThis);
        if (CodeBlock* codeBlock = executable->codeBlock())
            out.print(*codeBlock);
        else
            out.print("ModuleProgramExecutable w/o CodeBlock"_s);
        return;
    }
    case FunctionExecutableType: {
        FunctionExecutable* function = jsCast<FunctionExecutable*>(realThis);
        if (!function->eitherCodeBlock())
            out.print("FunctionExecutable w/o CodeBlock"_s);
        else {
            CommaPrinter comma("/"_s);
            if (function->codeBlockForCall())
                out.print(comma, *function->codeBlockForCall());
            if (function->codeBlockForConstruct())
                out.print(comma, *function->codeBlockForConstruct());
        }
        return;
    }
    default:
        RELEASE_ASSERT_NOT_REACHED();
    }
}

CodeBlockHash ExecutableBase::hashFor(CodeSpecializationKind kind) const
{
    if (type() == NativeExecutableType)
        return jsCast<const NativeExecutable*>(this)->hashFor(kind);
    
    return jsCast<const ScriptExecutable*>(this)->hashFor(kind);
}

} // namespace JSC
