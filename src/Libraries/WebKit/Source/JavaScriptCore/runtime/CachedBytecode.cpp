/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#include "CachedBytecode.h"

#include "CachedTypes.h"
#include "UnlinkedFunctionExecutable.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

void CachedBytecode::addGlobalUpdate(Ref<CachedBytecode> bytecode)
{
    ASSERT(m_updates.isEmpty());
    m_leafExecutables.clear();
    copyLeafExecutables(bytecode.get());
    m_updates.append(CacheUpdate::GlobalUpdate { WTFMove(bytecode->m_payload) });
}

void CachedBytecode::addFunctionUpdate(const UnlinkedFunctionExecutable* executable, CodeSpecializationKind kind, Ref<CachedBytecode> bytecode)
{
    auto it = m_leafExecutables.find(executable);
    ASSERT(it != m_leafExecutables.end());
    ptrdiff_t offset = it->value.base();
    ASSERT(offset);
    copyLeafExecutables(bytecode.get());
    m_updates.append(CacheUpdate::FunctionUpdate { offset, kind, { executable->features(), executable->lexicallyScopedFeatures(), executable->hasCapturedVariables() }, WTFMove(bytecode->m_payload) });
}

void CachedBytecode::copyLeafExecutables(const CachedBytecode& bytecode)
{
    for (const auto& it : bytecode.m_leafExecutables) {
        auto addResult = m_leafExecutables.add(it.key, it.value + m_size);
        ASSERT_UNUSED(addResult, addResult.isNewEntry);
    }
    m_size += bytecode.size();
}

void CachedBytecode::commitUpdates(const ForEachUpdateCallback& callback) const
{
    off_t offset = m_payload.size();
    for (const auto& update : m_updates) {
        const CachePayload* payload = nullptr;
        if (update.isGlobal())
            payload = &update.asGlobal().m_payload;
        else {
            const CacheUpdate::FunctionUpdate& functionUpdate = update.asFunction();
            payload = &functionUpdate.m_payload;
            {
                ptrdiff_t kindOffset = functionUpdate.m_kind == CodeForCall ? CachedFunctionExecutableOffsets::codeBlockForCallOffset() : CachedFunctionExecutableOffsets::codeBlockForConstructOffset();
                ptrdiff_t codeBlockOffset = functionUpdate.m_base + kindOffset + CachedWriteBarrierOffsets::ptrOffset() + CachedPtrOffsets::offsetOffset();
                ptrdiff_t offsetPayload = static_cast<ptrdiff_t>(offset) - codeBlockOffset;
                static_assert(std::is_same<decltype(VariableLengthObjectBase::m_offset), ptrdiff_t>::value);
                callback(codeBlockOffset, { reinterpret_cast<const uint8_t*>(&offsetPayload), sizeof(ptrdiff_t) });
            }

            {
                ptrdiff_t metadataOffset = functionUpdate.m_base + CachedFunctionExecutableOffsets::metadataOffset();
                callback(metadataOffset, { reinterpret_cast<const uint8_t*>(&functionUpdate.m_metadata), sizeof(functionUpdate.m_metadata) });
            }
        }

        ASSERT(payload);
        callback(offset, payload->span());
        offset += payload->size();
    }
    ASSERT(static_cast<size_t>(offset) == m_size);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
