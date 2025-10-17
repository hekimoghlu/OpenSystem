/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
#include "config.h"
#include "CSSGridTemplateAreasValue.h"

#include "GridArea.h"
#include <wtf/FixedVector.h>
#include <wtf/HashSet.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

CSSGridTemplateAreasValue::CSSGridTemplateAreasValue(NamedGridAreaMap map, size_t rowCount, size_t columnCount)
    : CSSValue(ClassType::GridTemplateAreas)
    , m_map(WTFMove(map))
    , m_rowCount(rowCount)
    , m_columnCount(columnCount)
{
    ASSERT(m_rowCount);
    ASSERT(m_columnCount);
}

Ref<CSSGridTemplateAreasValue> CSSGridTemplateAreasValue::create(NamedGridAreaMap map, size_t rowCount, size_t columnCount)
{
    return adoptRef(*new CSSGridTemplateAreasValue(WTFMove(map), rowCount, columnCount));
}

static String stringForPosition(const NamedGridAreaMap& gridAreaMap, size_t row, size_t column)
{
    UncheckedKeyHashSet<String> candidates;
    for (auto& it : gridAreaMap.map) {
        auto& area = it.value;
        if (row >= area.rows.startLine() && row < area.rows.endLine())
            candidates.add(it.key);
    }
    for (auto& it : gridAreaMap.map) {
        auto& area = it.value;
        if (column >= area.columns.startLine() && column < area.columns.endLine() && candidates.contains(it.key))
            return it.key;
    }
    return "."_s;
}

String CSSGridTemplateAreasValue::stringForRow(size_t row) const
{
    FixedVector<String> columns(m_columnCount);
    for (auto& it : m_map.map) {
        auto& area = it.value;
        if (row >= area.rows.startLine() && row < area.rows.endLine()) {
            for (unsigned i = area.columns.startLine(); i < area.columns.endLine(); i++)
                columns[i] = it.key;
        }
    }
    StringBuilder builder;
    bool first = true;
    for (auto& name : columns) {
        if (!first)
            builder.append(' ');
        first = false;
        if (name.isNull())
            builder.append('.');
        else
            builder.append(name);
    }
    return builder.toString();
}

String CSSGridTemplateAreasValue::customCSSText() const
{
    StringBuilder builder;
    for (size_t row = 0; row < m_rowCount; ++row) {
        builder.append('"');
        for (size_t column = 0; column < m_columnCount; ++column) {
            builder.append(stringForPosition(m_map, row, column));
            if (column != m_columnCount - 1)
                builder.append(' ');
        }
        builder.append('"');
        if (row != m_rowCount - 1)
            builder.append(' ');
    }
    return builder.toString();
}

bool CSSGridTemplateAreasValue::equals(const CSSGridTemplateAreasValue& other) const
{
    return m_map.map == other.m_map.map && m_rowCount == other.m_rowCount && m_columnCount == other.m_columnCount;
}

} // namespace WebCore
