/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 17, 2022.
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

#include "UnlinkedCodeBlock.h"

namespace JSC {

class CachedFunctionCodeBlock;

class UnlinkedFunctionCodeBlock final : public UnlinkedCodeBlock {
public:
    typedef UnlinkedCodeBlock Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.unlinkedFunctionCodeBlockSpace<mode>();
    }

    static UnlinkedFunctionCodeBlock* create(VM& vm, CodeType codeType, const ExecutableInfo& info, OptionSet<CodeGenerationMode> codeGenerationMode)
    {
        UnlinkedFunctionCodeBlock* instance = new (NotNull, allocateCell<UnlinkedFunctionCodeBlock>(vm)) UnlinkedFunctionCodeBlock(vm, vm.unlinkedFunctionCodeBlockStructure.get(), codeType, info, codeGenerationMode);
        instance->finishCreation(vm);
        return instance;
    }

    static void destroy(JSCell*);

private:
    friend CachedFunctionCodeBlock;

    UnlinkedFunctionCodeBlock(VM& vm, Structure* structure, CodeType codeType, const ExecutableInfo& info, OptionSet<CodeGenerationMode> codeGenerationMode)
        : Base(vm, structure, codeType, info, codeGenerationMode)
    {
    }

    UnlinkedFunctionCodeBlock(Decoder&, const CachedFunctionCodeBlock&);
    
public:
    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;
};

}
