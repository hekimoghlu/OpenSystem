/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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
#include "InlineCallFrame.h"

#include "CodeBlock.h"
#include "JSCJSValueInlines.h"

namespace JSC {

DEFINE_COMPACT_ALLOCATOR_WITH_HEAP_IDENTIFIER(InlineCallFrame);

JSFunction* InlineCallFrame::calleeConstant() const
{
    if (calleeRecovery.isConstant())
        return jsCast<JSFunction*>(calleeRecovery.constant());
    return nullptr;
}

JSFunction* InlineCallFrame::calleeForCallFrame(CallFrame* callFrame) const
{
    return jsCast<JSFunction*>(calleeRecovery.recover(callFrame));
}

CodeBlockHash InlineCallFrame::hash() const
{
    return baselineCodeBlock->hash();
}

CString InlineCallFrame::hashAsStringIfPossible() const
{
    return baselineCodeBlock->hashAsStringIfPossible();
}

CString InlineCallFrame::inferredName() const
{
    return jsCast<FunctionExecutable*>(baselineCodeBlock->ownerExecutable())->ecmaName().utf8();
}

void InlineCallFrame::dumpBriefFunctionInformation(PrintStream& out) const
{
    out.print(inferredName(), "#", hashAsStringIfPossible());
}

void InlineCallFrame::dumpInContext(PrintStream& out, DumpContext* context) const
{
    out.print(briefFunctionInformation(), ":<", RawPointer(baselineCodeBlock.get()));
    if (isInStrictContext())
        out.print(" (StrictMode)");
    out.print(", ", directCaller.bytecodeIndex(), ", ", static_cast<Kind>(kind));
    if (isClosureCall)
        out.print(", closure call");
    else
        out.print(", known callee: ", inContext(calleeRecovery.constant(), context));
    out.print(", numArgs+this = ", argumentCountIncludingThis);
    out.print(", numFixup = ", m_argumentsWithFixup.size() - argumentCountIncludingThis);
    out.print(", stackOffset = ", stackOffset);
    out.print(" (", virtualRegisterForLocal(0), " maps to ", virtualRegisterForLocal(0) + stackOffset, ")>");
}

void InlineCallFrame::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

} // namespace JSC

namespace WTF {

void printInternal(PrintStream& out, JSC::InlineCallFrame::Kind kind)
{
    switch (kind) {
    case JSC::InlineCallFrame::Call:
        out.print("Call");
        return;
    case JSC::InlineCallFrame::Construct:
        out.print("Construct");
        return;
    case JSC::InlineCallFrame::TailCall:
        out.print("TailCall");
        return;
    case JSC::InlineCallFrame::CallVarargs:
        out.print("CallVarargs");
        return;
    case JSC::InlineCallFrame::ConstructVarargs:
        out.print("ConstructVarargs");
        return;
    case JSC::InlineCallFrame::TailCallVarargs:
        out.print("TailCallVarargs");
        return;
    case JSC::InlineCallFrame::GetterCall:
        out.print("GetterCall");
        return;
    case JSC::InlineCallFrame::SetterCall:
        out.print("SetterCall");
        return;
    case JSC::InlineCallFrame::ProxyObjectLoadCall:
        out.print("ProxyObjectLoadCall");
        return;
    case JSC::InlineCallFrame::ProxyObjectStoreCall:
        out.print("ProxyObjectStoreCall");
        return;
    case JSC::InlineCallFrame::ProxyObjectInCall:
        out.print("ProxyObjectInCall");
        return;
    case JSC::InlineCallFrame::BoundFunctionCall:
        out.print("BoundFunctionCall");
        return;
    case JSC::InlineCallFrame::BoundFunctionTailCall:
        out.print("BoundFunctionTailCall");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

