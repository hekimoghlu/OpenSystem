/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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

#if ENABLE(WEBASSEMBLY)

#include "WasmTypeDefinition.h"
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace Wasm {

class Tag final : public ThreadSafeRefCounted<Tag> {
    WTF_MAKE_TZONE_ALLOCATED(Tag);
    WTF_MAKE_NONCOPYABLE(Tag);
public:
    static Ref<Tag> create(Ref<const TypeDefinition>&& type) { return adoptRef(*new Tag(WTFMove(type))); }

    FunctionArgCount parameterCount() const { return m_type->as<FunctionSignature>()->argumentCount(); }

    size_t parameterBufferSize() const
    {
        size_t result = 0;
        for (size_t i = 0; i < parameterCount(); i ++)
            result += m_type->as<FunctionSignature>()->argumentType(i).kind == TypeKind::V128 ? 2 : 1;
        return result;
    }

    Type parameter(FunctionArgCount i) const { return m_type->as<FunctionSignature>()->argumentType(i); }
    TypeIndex typeIndex() const { return m_type->index(); }

    // Since (1) we do not copy Wasm::Tag and (2) we always allocate Wasm::Tag from heap, we can use
    // pointer comparison for identity check.
    bool operator==(const Tag& other) const { return this == &other; }

    const FunctionSignature& type() const { return *m_type->as<FunctionSignature>(); }

    static Tag& jsExceptionTag();

private:
    Tag(Ref<const TypeDefinition>&& type)
        : m_type(WTFMove(type))
    {
        ASSERT(m_type->is<FunctionSignature>());
    }

    Ref<const TypeDefinition> m_type;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
