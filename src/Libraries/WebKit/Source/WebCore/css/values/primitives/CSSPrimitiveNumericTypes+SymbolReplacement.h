/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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

#include "CSSPrimitiveNumericTypes.h"
#include "CSSSymbol.h"
#include <wtf/Brigand.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

class CSSCalcSymbolTable;

namespace CSS {

// MARK: Add/Remove Symbol

// MARK: Transform CSS type list -> CSS type list + Symbol

// Transform `brigand::list<css1, css2, ...>`  -> `brigand::list<css1, css2, ..., symbol>`
template<typename TypeList> struct PlusSymbolLazy {
    using type = brigand::append<TypeList, brigand::list<Symbol>>;
};
template<typename TypeList> using PlusSymbol = typename PlusSymbolLazy<TypeList>::type;

// MARK: Transform CSS type list + Symbol -> CSS type list - Symbol

// Transform `brigand::list<css1, css2, ..., symbol>`  -> `brigand::list<css1, css2, ...>`
template<typename TypeList> struct MinusSymbolLazy {
    using type = brigand::remove_if<TypeList, IsSymbol<brigand::_1>>;
};
template<typename TypeList> using MinusSymbol = typename MinusSymbolLazy<TypeList>::type;

// MARK: - Symbol replacement

// Replaces the symbol with a value from the symbol table. This is only relevant
// for Symbol, so a catchall overload that implements the identity function is
// provided to allow generic replacement.
Number<> replaceSymbol(Symbol, const CSSCalcSymbolTable&);

template<typename T> constexpr auto replaceSymbol(T value, const CSSCalcSymbolTable&) -> T
{
    return value;
}

template<typename... Ts> using TypesMinusSymbol = VariantOrSingle<MinusSymbol<brigand::list<Ts...>>>;

template<typename... Ts> constexpr auto replaceSymbol(const std::variant<Ts...>& component, const CSSCalcSymbolTable& symbolTable) -> TypesMinusSymbol<Ts...>
{
    return WTF::switchOn(component, [&](auto part) -> TypesMinusSymbol<Ts...> { return replaceSymbol(part, symbolTable); });
}

template<typename T> constexpr decltype(auto) replaceSymbol(const std::optional<T>& component, const CSSCalcSymbolTable& symbolTable)
{
    return component ? std::make_optional(replaceSymbol(*component, symbolTable)) : std::nullopt;
}

} // namespace CSS
} // namespace WebCore
