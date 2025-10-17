/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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

#include "CacheableIdentifier.h"
#include "StructureID.h"
#include "VM.h"
#include <wtf/FixedVector.h>
#include <wtf/Vector.h>

namespace JSC {

class JSCell;
class JSGlobalObject;
class JSObject;
class PropertySlot;
class Structure;

class PolyProtoAccessChain final : public ThreadSafeRefCounted<PolyProtoAccessChain> {
public:
    // Returns nullptr when invalid.
    static RefPtr<PolyProtoAccessChain> tryCreate(JSGlobalObject*, JSCell* base, CacheableIdentifier, const PropertySlot&);
    static RefPtr<PolyProtoAccessChain> tryCreate(JSGlobalObject*, JSCell* base, CacheableIdentifier, JSObject* target);

    const FixedVector<StructureID>& chain() const { return m_chain; }

    void dump(Structure* baseStructure, PrintStream& out) const;

    bool operator==(const PolyProtoAccessChain&) const;

    bool needImpurePropertyWatchpoint(VM&) const;

    template <typename Func>
    void forEach(VM&, Structure* baseStructure, const Func& func) const
    {
        bool atEnd = !m_chain.size();
        func(baseStructure, atEnd);
        for (unsigned i = 0; i < m_chain.size(); ++i) {
            atEnd = i + 1 == m_chain.size();
            func(m_chain[i].decode(), atEnd);
        }
    }

    Structure* slotBaseStructure(VM&, Structure* baseStructure) const
    {
        if (m_chain.size())
            return m_chain.last().decode();
        return baseStructure;
    }

private:
    explicit PolyProtoAccessChain(Vector<StructureID>&& chain)
        : m_chain(WTFMove(chain))
    {
    }

    // This does not include the base. We rely on AccessCase providing it for us. That said, this data
    // structure is tied to the base that it was created with.
    FixedVector<StructureID> m_chain;
};

}
