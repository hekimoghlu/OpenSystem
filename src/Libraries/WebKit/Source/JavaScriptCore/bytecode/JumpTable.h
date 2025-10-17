/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

#include "CodeLocation.h"
#include "UnlinkedCodeBlock.h"
#include <wtf/FixedVector.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/StringImpl.h>

namespace JSC {

#if ENABLE(JIT)
    struct StringJumpTable {
        FixedVector<CodeLocationLabel<JSSwitchPtrTag>> m_ctiOffsets;

        void ensureCTITable(const UnlinkedStringJumpTable& unlinkedTable)
        {
            if (!isEmpty())
                return;
            m_ctiOffsets = FixedVector<CodeLocationLabel<JSSwitchPtrTag>>(unlinkedTable.m_offsetTable.size() + 1);
        }

        inline CodeLocationLabel<JSSwitchPtrTag> ctiForValue(const UnlinkedStringJumpTable& unlinkedTable, StringImpl* value) const
        {
            auto loc = unlinkedTable.m_offsetTable.find(value);
            if (loc == unlinkedTable.m_offsetTable.end())
                return m_ctiOffsets[unlinkedTable.m_offsetTable.size()];
            return m_ctiOffsets[loc->value.m_indexInTable];
        }

        CodeLocationLabel<JSSwitchPtrTag> ctiDefault() const { return m_ctiOffsets.last(); }

        bool isEmpty() const { return m_ctiOffsets.isEmpty(); }
    };

    struct SimpleJumpTable {
        FixedVector<CodeLocationLabel<JSSwitchPtrTag>> m_ctiOffsets;
        CodeLocationLabel<JSSwitchPtrTag> m_ctiDefault;

        void ensureCTITable(const UnlinkedSimpleJumpTable& unlinkedTable)
        {
            if (!isEmpty())
                return;
            m_ctiOffsets = FixedVector<CodeLocationLabel<JSSwitchPtrTag>>(unlinkedTable.m_branchOffsets.size());
        }

        inline CodeLocationLabel<JSSwitchPtrTag> ctiForValue(int32_t min, int32_t value) const
        {
            if (value >= min && static_cast<uint32_t>(value - min) < m_ctiOffsets.size())
                return m_ctiOffsets[value - min];
            return m_ctiDefault;
        }

        bool isEmpty() const { return m_ctiOffsets.isEmpty(); }
    };
#endif

} // namespace JSC
