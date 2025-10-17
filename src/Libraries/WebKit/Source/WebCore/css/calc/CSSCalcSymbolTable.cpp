/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#include "CSSCalcSymbolTable.h"

#include "CSSUnits.h"

namespace WebCore {

CSSCalcSymbolTable::CSSCalcSymbolTable(std::initializer_list<std::tuple<CSSValueID, CSSUnitType, double>> initializer)
{
    for (auto& [identifier, type, value] : initializer)
        m_table.add(identifier, std::make_pair(type, value));
}

std::optional<CSSCalcSymbolTable::Value> CSSCalcSymbolTable::get(CSSValueID valueID) const
{
    auto it = m_table.find(valueID);
    if (it == m_table.end())
        return std::nullopt;

    return {{ it->value.first, it->value.second }};
}

bool CSSCalcSymbolTable::contains(CSSValueID valueID) const
{
    return m_table.contains(valueID);
}

} // namespace WebCore
