/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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

#include "FunctionExecutable.h"
#include "InferredValueInlines.h"
#include "ScriptExecutableInlines.h"

namespace JSC {

inline Structure* FunctionExecutable::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
{
    return Structure::create(vm, globalObject, proto, TypeInfo(FunctionExecutableType, StructureFlags), info());
}

inline void FunctionExecutable::finalizeUnconditionally(VM& vm, CollectionScope collectionScope)
{
    m_singleton.finalizeUnconditionally(vm, collectionScope);
    finalizeCodeBlockEdge(vm, m_codeBlockForCall);
    finalizeCodeBlockEdge(vm, m_codeBlockForConstruct);
    vm.heap.functionExecutableSpaceAndSet.outputConstraintsSet.remove(this);
}

inline FunctionCodeBlock* FunctionExecutable::replaceCodeBlockWith(VM& vm, CodeSpecializationKind kind, CodeBlock* newCodeBlock)
{
    if (kind == CodeForCall) {
        FunctionCodeBlock* oldCodeBlock = codeBlockForCall();
        m_codeBlockForCall.setMayBeNull(vm, this, newCodeBlock);
        return oldCodeBlock;
    }
    ASSERT(kind == CodeForConstruct);
    FunctionCodeBlock* oldCodeBlock = codeBlockForConstruct();
    m_codeBlockForConstruct.setMayBeNull(vm, this, newCodeBlock);
    return oldCodeBlock;
}

inline JSString* FunctionExecutable::toString(JSGlobalObject* globalObject)
{
    RareData& rareData = ensureRareData();
    if (!rareData.m_asString)
        return toStringSlow(globalObject);
    return rareData.m_asString.get();
}

} // namespace JSC

