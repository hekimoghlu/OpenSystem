/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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

#include "Structure.h"
#include "Symbol.h"
#include "Watchpoint.h"
#include "WriteBarrierInlines.h"

namespace WTF {

class UniquedStringImpl;

} // namespace WTF

namespace JSC {

class BrandedStructure final : public Structure {
    typedef Structure Base;

public:

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.brandedStructureSpace();
    }

    ALWAYS_INLINE bool checkBrand(Symbol* brand)
    {
        UniquedStringImpl* brandUid = &brand->uid();
        for (BrandedStructure* currentStructure = this; currentStructure; currentStructure = jsCast<BrandedStructure*>(currentStructure->m_parentBrand.get())) {
            if (brandUid == currentStructure->m_brand)
                return true;
        }
        return false;
    }

    template<typename Visitor>
    static void visitAdditionalChildren(JSCell* cell, Visitor& visitor)
    {
        BrandedStructure* thisObject = jsCast<BrandedStructure*>(cell);
        visitor.append(thisObject->m_parentBrand);
    }

private:
    BrandedStructure(VM&, Structure*, UniquedStringImpl* brand);
    BrandedStructure(VM&, BrandedStructure*);

    static Structure* create(VM&, Structure*, UniquedStringImpl* brand, DeferredStructureTransitionWatchpointFire* = nullptr);

    void destruct()
    {
        m_brand = nullptr;
    }

    CompactRefPtr<UniquedStringImpl> m_brand;
    WriteBarrierStructureID m_parentBrand;

    friend class Structure;
};

} // namespace JSC
