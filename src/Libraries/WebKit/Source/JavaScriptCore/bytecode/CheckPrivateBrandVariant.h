/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "StructureSet.h"
#include <wtf/Box.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class CheckPrivateBrandStatus;

class CheckPrivateBrandVariant {
    WTF_MAKE_TZONE_ALLOCATED(CheckPrivateBrandVariant);
public:
    CheckPrivateBrandVariant(CacheableIdentifier, const StructureSet& = StructureSet());

    ~CheckPrivateBrandVariant();

    const StructureSet& structureSet() const { return m_structureSet; }
    StructureSet& structureSet() { return m_structureSet; }

    bool attemptToMerge(const CheckPrivateBrandVariant& other);

    DECLARE_VISIT_AGGREGATE;
    template<typename Visitor> void markIfCheap(Visitor&);
    bool finalize(VM&);

    void dump(PrintStream&) const;
    void dumpInContext(PrintStream&, DumpContext*) const;

    CacheableIdentifier identifier() const { return m_identifier; }

    bool overlaps(const CheckPrivateBrandVariant& other)
    {
        if (!!m_identifier != !!other.m_identifier)
            return true;
        if (m_identifier) {
            if (m_identifier != other.m_identifier)
                return false;
        }
        return structureSet().overlaps(other.structureSet());
    }

private:
    friend class CheckPrivateBrandStatus;

    StructureSet m_structureSet;
    CacheableIdentifier m_identifier;
};

} // namespace JSC
