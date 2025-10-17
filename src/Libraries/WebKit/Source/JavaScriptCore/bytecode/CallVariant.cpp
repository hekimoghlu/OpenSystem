/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#include "CallVariant.h"

#include "JSCInlines.h"
#include <wtf/ListDump.h>

namespace JSC {

bool CallVariant::finalize(VM& vm)
{
    if (m_callee && !vm.heap.isMarked(m_callee))
        return false;
    return true;
}

bool CallVariant::merge(const CallVariant& other)
{
    if (*this == other)
        return true;
    if (executable() == other.executable()) {
        *this = despecifiedClosure();
        return true;
    }
    return false;
}

void CallVariant::filter(JSValue value)
{
    if (!*this)
        return;
    
    if (!isClosureCall()) {
        if (nonExecutableCallee() != value)
            *this = CallVariant();
        return;
    }
    
    if (JSFunction* function = jsDynamicCast<JSFunction*>(value)) {
        if (function->executable() == executable())
            *this = CallVariant(function);
        else
            *this = CallVariant();
        return;
    }
    
    *this = CallVariant();
}

void CallVariant::dump(PrintStream& out) const
{
    if (!*this) {
        out.print("null");
        return;
    }
    
    if (InternalFunction* internalFunction = this->internalFunction()) {
        out.print("InternalFunction: ", JSValue(internalFunction));
        return;
    }
    
    if (JSFunction* function = this->function()) {
        out.print("(Function: ", JSValue(function), "; Executable: ", *executable(), ")");
        return;
    }
    
    if (ExecutableBase* executable = this->executable()) {
        out.print("(Executable: ", *executable, ")");
        return;
    }

    out.print("Non-executable callee: ", *nonExecutableCallee());
}

CallVariantList variantListWithVariant(const CallVariantList& list, CallVariant variantToAdd)
{
    ASSERT(variantToAdd);
    CallVariantList result;
    for (CallVariant variant : list) {
        ASSERT(variant);
        if (!!variantToAdd) {
            if (variant == variantToAdd)
                variantToAdd = CallVariant();
            else if (variant.despecifiedClosure() == variantToAdd.despecifiedClosure()) {
                variant = variant.despecifiedClosure();
                variantToAdd = CallVariant();
            }
        }
        result.append(variant);
    }
    if (!!variantToAdd)
        result.append(variantToAdd);
    
    if (ASSERT_ENABLED) {
        for (unsigned i = 0; i < result.size(); ++i) {
            for (unsigned j = i + 1; j < result.size(); ++j) {
                if (result[i] != result[j])
                    continue;
                
                dataLog("variantListWithVariant(", listDump(list), ", ", variantToAdd, ") failed: got duplicates in result: ", listDump(result), "\n");
                RELEASE_ASSERT_NOT_REACHED();
            }
        }
    }
    
    return result;
}

CallVariantList despecifiedVariantList(const CallVariantList& list)
{
    CallVariantList result;
    for (CallVariant variant : list)
        result = variantListWithVariant(result, variant.despecifiedClosure());
    return result;
}

} // namespace JSC

