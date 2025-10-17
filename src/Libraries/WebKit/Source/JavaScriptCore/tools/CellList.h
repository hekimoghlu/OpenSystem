/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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

#include "CellProfile.h"
#include <wtf/HashMap.h>
#include <wtf/SegmentedVector.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class CellList {
    WTF_MAKE_TZONE_ALLOCATED(CellList);
public:
    CellList(const char* name)
        : m_name(name)
    {
    }
    
    const char* name() const { return m_name; }
    size_t size() const { return m_cells.size(); }

    typedef SegmentedVector<CellProfile, 64> CellProfileVector;
    CellProfileVector& cells() { return m_cells; }

    void add(CellProfile&& profile)
    {
        m_cells.append(WTFMove(profile));
        m_mapIsUpToDate = false;
    }

    void reset();

    CellProfile* find(HeapCell*);

private:
    const char* m_name;
    CellProfileVector m_cells;

    bool m_mapIsUpToDate { false };
    UncheckedKeyHashMap<HeapCell*, CellProfile*> m_map;
};
    
} // namespace JSC
