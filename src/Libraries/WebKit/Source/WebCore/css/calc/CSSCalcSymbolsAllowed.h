/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 30, 2024.
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

#include "CSSValueKeywords.h"
#include <optional>
#include <wtf/HashMap.h>

namespace WebCore {

enum class CSSUnitType : uint8_t;

class CSSCalcSymbolsAllowed {
public:
    CSSCalcSymbolsAllowed() = default;
    CSSCalcSymbolsAllowed(std::initializer_list<std::tuple<CSSValueID, CSSUnitType>>);

    CSSCalcSymbolsAllowed& operator=(const CSSCalcSymbolsAllowed&) = default;
    CSSCalcSymbolsAllowed(const CSSCalcSymbolsAllowed&) = default;
    CSSCalcSymbolsAllowed& operator=(CSSCalcSymbolsAllowed&&) = default;
    CSSCalcSymbolsAllowed(CSSCalcSymbolsAllowed&&) = default;

    std::optional<CSSUnitType> get(CSSValueID) const;
    bool contains(CSSValueID) const;

private:
    // FIXME: A UncheckedKeyHashMap here is not ideal, as these tables are always constant expressions
    // and always quite small (currently always 4, but in the future will include one that
    // is 5 elements, hard coding a size of 4 would be unfortunate. A more ideal solution
    // would be to have this be a SortedArrayMap, but it currently has the restriction that
    // that both the type and size of the storage is fixed. I can probably be updated to
    // use the size only at construction and store store a std::span instead (or a variant
    // version could be made).

    UncheckedKeyHashMap<CSSValueID, CSSUnitType> m_table;
};

}
