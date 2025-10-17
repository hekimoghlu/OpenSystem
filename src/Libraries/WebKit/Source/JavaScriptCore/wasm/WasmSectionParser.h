/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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

#include "WasmFormat.h"
#include "WasmOps.h"
#include "WasmParser.h"
#include <wtf/text/ASCIILiteral.h>
#include <wtf/text/MakeString.h>

namespace JSC { namespace Wasm {

class SectionParser final : public Parser<void> {
public:
    SectionParser(std::span<const uint8_t> data, size_t offsetInSource, ModuleInformation& info)
        : Parser(data)
        , m_offsetInSource(offsetInSource)
        , m_info(info)
    {
    }

#define WASM_SECTION_DECLARE_PARSER(NAME, ID, ORDERING, DESCRIPTION) PartialResult WARN_UNUSED_RETURN parse ## NAME();
    FOR_EACH_KNOWN_WASM_SECTION(WASM_SECTION_DECLARE_PARSER)
#undef WASM_SECTION_DECLARE_PARSER

    PartialResult WARN_UNUSED_RETURN parseCustom();

private:
    template <typename ...Args>
    NEVER_INLINE UnexpectedResult WARN_UNUSED_RETURN fail(Args... args) const
    {
        using namespace FailureHelper; // See ADL comment in namespace above.
        if (UNLIKELY(ASSERT_ENABLED && Options::crashOnFailedWasmValidate()))
            CRASH();

        return UnexpectedResult(makeString("WebAssembly.Module doesn't parse at byte "_s, String::number(m_offset + m_offsetInSource), ": "_s, makeString(args)...));
    }

    PartialResult WARN_UNUSED_RETURN parseGlobalType(GlobalInformation&);
    PartialResult WARN_UNUSED_RETURN parseMemoryHelper(bool isImport);
    PartialResult WARN_UNUSED_RETURN parseTableHelper(bool isImport);
    enum class LimitsType { Memory, Table };
    PartialResult WARN_UNUSED_RETURN parseResizableLimits(uint32_t& initial, std::optional<uint32_t>& maximum, bool& isShared, LimitsType);
    PartialResult WARN_UNUSED_RETURN parseInitExpr(uint8_t&, bool&, uint64_t&, v128_t&, Type, Type& initExprType);
    PartialResult WARN_UNUSED_RETURN parseI32InitExpr(std::optional<I32InitExpr>&, ASCIILiteral failMessage);

    PartialResult WARN_UNUSED_RETURN parseFunctionType(uint32_t position, RefPtr<TypeDefinition>&);
    PartialResult WARN_UNUSED_RETURN parsePackedType(PackedType&);
    PartialResult WARN_UNUSED_RETURN parseStorageType(StorageType&);
    PartialResult WARN_UNUSED_RETURN parseStructType(uint32_t position, RefPtr<TypeDefinition>&);
    PartialResult WARN_UNUSED_RETURN parseArrayType(uint32_t position, RefPtr<TypeDefinition>&);
    PartialResult WARN_UNUSED_RETURN parseRecursionGroup(uint32_t position, RefPtr<TypeDefinition>&);
    PartialResult WARN_UNUSED_RETURN parseSubtype(uint32_t position, RefPtr<TypeDefinition>&, Vector<TypeIndex>&, bool);

    PartialResult WARN_UNUSED_RETURN validateElementTableIdx(uint32_t, Type);
    PartialResult WARN_UNUSED_RETURN parseI32InitExprForElementSection(std::optional<I32InitExpr>&);
    PartialResult WARN_UNUSED_RETURN parseElementKind(uint8_t& elementKind);
    PartialResult WARN_UNUSED_RETURN parseIndexCountForElementSection(uint32_t&, const unsigned);
    PartialResult WARN_UNUSED_RETURN parseElementSegmentVectorOfExpressions(Type, Vector<Element::InitializationType>&, Vector<uint64_t>&, const unsigned, const unsigned);
    PartialResult WARN_UNUSED_RETURN parseElementSegmentVectorOfIndexes(Vector<Element::InitializationType>&, Vector<uint64_t>&, const unsigned, const unsigned);

    PartialResult WARN_UNUSED_RETURN parseI32InitExprForDataSection(std::optional<I32InitExpr>&);

    static bool checkStructuralSubtype(const TypeDefinition&, const TypeDefinition&);
    PartialResult WARN_UNUSED_RETURN checkSubtypeValidity(const TypeDefinition&);

    size_t m_offsetInSource;
    Ref<ModuleInformation> m_info;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
