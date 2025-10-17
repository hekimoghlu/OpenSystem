/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "WasmIndexOrName.h"

#include <wtf/PrintStream.h>
#include <wtf/text/MakeString.h>

namespace JSC { namespace Wasm {

IndexOrName::IndexOrName(Index index, std::pair<const Name*, RefPtr<NameSection>>&& name)
{
#if USE(JSVALUE64)
    static_assert(sizeof(m_indexName.index) == sizeof(m_indexName.name), "bit-tagging depends on sizes being equal");
    ASSERT(!(index & allTags));
    ASSERT(!(std::bit_cast<Index>(name.first) & allTags));
    if (name.first)
        m_indexName.name = name.first;
    else
        m_indexName.index = indexTag | index;
#elif USE(JSVALUE32_64)
    if (name.first) {
        m_indexName.name = name.first;
        m_kind = Kind::Name;
    } else {
        m_indexName.index = index;
        m_kind = Kind::Index;
    }
#endif
    m_nameSection = WTFMove(name.second);
}

void IndexOrName::dump(PrintStream& out) const
{
    if (isEmpty() || !nameSection()) {
        out.print("wasm-stub"_s);
        if (isIndex())
            out.print('[', index(), ']');
        return;
    }

    auto moduleName = nameSection()->moduleName.size() ? nameSection()->moduleName.span() : nameSection()->moduleHash.span();
    if (isIndex())
        out.print(moduleName, ".wasm-function["_s, index(), "]");
    else
        out.print(moduleName, ".wasm-function["_s, name()->span(), "]");
}

String makeString(const IndexOrName& ion)
{
    if (ion.isEmpty())
        return "wasm-stub"_s;
    auto moduleName = ion.nameSection()->moduleName.size() ? ion.nameSection()->moduleName.span() : ion.nameSection()->moduleHash.span();
    if (ion.isIndex())
        return makeString(moduleName, ".wasm-function["_s, ion.index(), ']');
    return makeString(moduleName, ".wasm-function["_s, ion.name()->span(), ']');
}

} } // namespace JSC::Wasm
