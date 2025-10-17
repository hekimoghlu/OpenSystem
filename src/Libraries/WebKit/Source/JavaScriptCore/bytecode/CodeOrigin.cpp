/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 24, 2023.
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
#include "CodeOrigin.h"

#include "CodeBlock.h"
#include "InlineCallFrame.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CodeOrigin::OutOfLineCodeOrigin);

unsigned CodeOrigin::inlineDepth() const
{
    unsigned result = 1;
    for (InlineCallFrame* current = inlineCallFrame(); current; current = current->directCaller.inlineCallFrame())
        result++;
    return result;
}

bool CodeOrigin::isApproximatelyEqualTo(const CodeOrigin& other, InlineCallFrame* terminal) const
{
    CodeOrigin a = *this;
    CodeOrigin b = other;

    if (!a.isSet())
        return !b.isSet();
    if (!b.isSet())
        return false;
    
    if (a.isHashTableDeletedValue())
        return b.isHashTableDeletedValue();
    if (b.isHashTableDeletedValue())
        return false;
    
    for (;;) {
        ASSERT(a.isSet());
        ASSERT(b.isSet());
        
        if (a.bytecodeIndex() != b.bytecodeIndex())
            return false;

        auto* aInlineCallFrame = a.inlineCallFrame();
        auto* bInlineCallFrame = b.inlineCallFrame();
        bool aHasInlineCallFrame = !!aInlineCallFrame && aInlineCallFrame != terminal;
        bool bHasInlineCallFrame = !!bInlineCallFrame;
        if (aHasInlineCallFrame != bHasInlineCallFrame)
            return false;
        
        if (!aHasInlineCallFrame)
            return true;
        
        if (aInlineCallFrame->baselineCodeBlock.get() != bInlineCallFrame->baselineCodeBlock.get())
            return false;
        
        a = aInlineCallFrame->directCaller;
        b = bInlineCallFrame->directCaller;
    }
}

unsigned CodeOrigin::approximateHash(InlineCallFrame* terminal) const
{
    if (!isSet())
        return 0;
    if (isHashTableDeletedValue())
        return 1;
    
    unsigned result = 2;
    CodeOrigin codeOrigin = *this;
    for (;;) {
        result += codeOrigin.bytecodeIndex().asBits();

        auto* inlineCallFrame = codeOrigin.inlineCallFrame();

        if (!inlineCallFrame)
            return result;
        
        if (inlineCallFrame == terminal)
            return result;
        
        result += WTF::PtrHash<JSCell*>::hash(inlineCallFrame->baselineCodeBlock.get());
        
        codeOrigin = inlineCallFrame->directCaller;
    }
}

Vector<CodeOrigin> CodeOrigin::inlineStack() const
{
    Vector<CodeOrigin> result(inlineDepth());
    result.last() = *this;
    unsigned index = result.size() - 2;
    for (InlineCallFrame* current = inlineCallFrame(); current; current = current->directCaller.inlineCallFrame())
        result[index--] = current->directCaller;
    RELEASE_ASSERT(!result[0].inlineCallFrame());
    return result;
}

CodeBlock* CodeOrigin::codeOriginOwner() const
{
    auto* inlineCallFrame = this->inlineCallFrame();
    if (!inlineCallFrame)
        return nullptr;
    return inlineCallFrame->baselineCodeBlock.get();
}

int CodeOrigin::stackOffset() const
{
    auto* inlineCallFrame = this->inlineCallFrame();
    if (!inlineCallFrame)
        return 0;
    return inlineCallFrame->stackOffset;
}

void CodeOrigin::dump(PrintStream& out) const
{
    if (!isSet()) {
        out.print("<none>");
        return;
    }
    
    Vector<CodeOrigin> stack = inlineStack();
    for (unsigned i = 0; i < stack.size(); ++i) {
        if (i)
            out.print(" --> ");
        
        if (InlineCallFrame* frame = stack[i].inlineCallFrame()) {
            out.print(frame->briefFunctionInformation(), ":<", RawPointer(frame->baselineCodeBlock.get()), "> ");
            if (frame->isClosureCall)
                out.print("(closure) ");
        }
        
        out.print(stack[i].bytecodeIndex());
    }
}

void CodeOrigin::dumpInContext(PrintStream& out, DumpContext*) const
{
    dump(out);
}

} // namespace JSC
