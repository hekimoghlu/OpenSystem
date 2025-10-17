/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include <wtf/text/SymbolRegistry.h>

#include <wtf/text/SymbolImpl.h>

namespace WTF {

SymbolRegistry::SymbolRegistry(Type type)
    : m_symbolType(type)
{
}

SymbolRegistry::~SymbolRegistry()
{
    for (auto& key : m_table) {
        ASSERT(key->isSymbol());
        static_cast<SymbolImpl*>(key.get())->asRegisteredSymbolImpl()->clearSymbolRegistry();
    }
}

Ref<RegisteredSymbolImpl> SymbolRegistry::symbolForKey(const String& rep)
{
    auto addResult = m_table.add(rep.impl());
    if (!addResult.isNewEntry) {
        ASSERT(addResult.iterator->get()->isSymbol());
        return *static_cast<SymbolImpl*>(addResult.iterator->get())->asRegisteredSymbolImpl();
    }

    RefPtr<RegisteredSymbolImpl> symbol;
    if (m_symbolType == Type::PrivateSymbol)
        symbol = RegisteredSymbolImpl::createPrivate(*rep.impl(), *this);
    else
        symbol = RegisteredSymbolImpl::create(*rep.impl(), *this);

    *addResult.iterator = symbol;
    return symbol.releaseNonNull();
}

// When removing a registered symbol from the table, we know it's already the one in the table, so no need for a string equality check.
struct SymbolRegistryTableRemovalHashTranslator {
    static unsigned hash(const RegisteredSymbolImpl* key) { return key->hash(); }
    static bool equal(const RefPtr<StringImpl>& a, const RegisteredSymbolImpl* b) { return a.get() == b; }
};

void SymbolRegistry::remove(RegisteredSymbolImpl& uid)
{
    ASSERT(uid.symbolRegistry() == this);
    auto iterator = m_table.find<SymbolRegistryTableRemovalHashTranslator>(&uid);
    ASSERT_WITH_MESSAGE(iterator != m_table.end(), "The string being removed is registered in the string table of an other thread!");
    m_table.remove(iterator);
}

}
