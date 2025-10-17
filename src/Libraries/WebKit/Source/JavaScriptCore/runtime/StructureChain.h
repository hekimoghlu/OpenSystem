/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

#include "JSCast.h"
#include "JSObject.h"
#include "Structure.h"
#include <wtf/StdLibExtras.h>
#include <wtf/UniqueArray.h>

namespace JSC {

class LLIntOffsetsExtractor;
class Structure;

class StructureChain final : public JSCell {
    friend class JIT;

public:
    using Base = JSCell;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.structureChainSpace();
    }

    static StructureChain* create(VM&, JSObject*);
    StructureID* head() { return m_vector.get(); }
    DECLARE_VISIT_CHILDREN;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    friend class LLIntOffsetsExtractor;

    void finishCreation(VM&, JSObject* head);

    StructureChain(VM&, Structure*, StructureID*);
    AuxiliaryBarrier<StructureID*> m_vector;
};

} // namespace JSC
