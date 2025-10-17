/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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
#include "DFGDesiredIdentifiers.h"

#if ENABLE(DFG_JIT)

#include "CodeBlock.h"
#include "DFGCommonData.h"
#include "IdentifierInlines.h"

namespace JSC { namespace DFG {

DesiredIdentifiers::DesiredIdentifiers()
    : m_codeBlock(nullptr)
    , m_didProcessIdentifiers(false)
{
}

DesiredIdentifiers::DesiredIdentifiers(CodeBlock* codeBlock)
    : m_codeBlock(codeBlock)
    , m_didProcessIdentifiers(false)
{
}

DesiredIdentifiers::~DesiredIdentifiers() = default;

unsigned DesiredIdentifiers::numberOfIdentifiers()
{
    return m_codeBlock->numberOfIdentifiers() + m_addedIdentifiers.size();
}

unsigned DesiredIdentifiers::ensure(UniquedStringImpl* rep)
{
    if (!m_didProcessIdentifiers) {
        // Do this now instead of the constructor so that we don't pay the price on the main
        // thread. Also, not all compilations need to call ensure().
        for (unsigned index = m_codeBlock->numberOfIdentifiers(); index--;)
            m_identifierNumberForName.add(m_codeBlock->identifier(index).impl(), index);
        m_didProcessIdentifiers = true;
    }

    auto addResult = m_identifierNumberForName.add(rep, numberOfIdentifiers());
    unsigned result = addResult.iterator->value;
    if (addResult.isNewEntry) {
        m_addedIdentifiers.append(rep);
        ASSERT(at(result) == rep);
    }
    return result;
}

UniquedStringImpl* DesiredIdentifiers::at(unsigned index) const
{
    UniquedStringImpl* result;
    if (index < m_codeBlock->numberOfIdentifiers())
        result = m_codeBlock->identifier(index).impl();
    else
        result = m_addedIdentifiers[index - m_codeBlock->numberOfIdentifiers()];
    ASSERT(result->hasAtLeastOneRef());
    return result;
}

void DesiredIdentifiers::reallyAdd(VM& vm, CommonData* commonData)
{
    unsigned index = 0;
    FixedVector<Identifier> identifiers(m_addedIdentifiers.size());
    for (auto rep : m_addedIdentifiers) {
        ASSERT(rep->hasAtLeastOneRef());
        identifiers[index++] = Identifier::fromUid(vm, rep);
    }
    if (!identifiers.isEmpty()) {
        ConcurrentJSLocker locker(m_codeBlock->m_lock);
        ASSERT(commonData->m_dfgIdentifiers.isEmpty());
        commonData->m_dfgIdentifiers = WTFMove(identifiers);
    }
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

