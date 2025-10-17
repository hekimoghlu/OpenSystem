/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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

#include "JSObject.h"
#include "SourceCode.h"

namespace JSC {

class JSSourceCode final : public JSCell {
public:
    using Base = JSCell;

    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;
    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    DECLARE_EXPORT_INFO;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.sourceCodeSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSSourceCode* create(VM& vm, Structure* structure, SourceCode&& sourceCode)
    {
        auto* result = new (NotNull, allocateCell<JSSourceCode>(vm)) JSSourceCode(vm, structure, WTFMove(sourceCode));
        result->finishCreation(vm);
        return result;
    }

    static JSSourceCode* create(VM& vm, SourceCode&& sourceCode)
    {
        return create(vm, vm.sourceCodeStructure.get(), WTFMove(sourceCode));
    }

    const SourceCode& sourceCode() const
    {
        return m_sourceCode;
    }

    static void destroy(JSCell*);

private:
    JSSourceCode(VM& vm, Structure* structure, SourceCode&& sourceCode)
        : Base(vm, structure)
        , m_sourceCode(WTFMove(sourceCode))
    {
    }

    SourceCode m_sourceCode;
};

} // namespace JSC
