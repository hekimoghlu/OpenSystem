/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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
#include "CombinedFiltersAlphabet.h"

#if ENABLE(CONTENT_EXTENSIONS)

namespace WebCore {

namespace ContentExtensions {

struct TermCreatorInput {
    const Term& term;
    Vector<std::unique_ptr<Term>>& internedTermsStorage;
};

inline void add(Hasher& hasher, const TermCreatorInput& input)
{
    add(hasher, input.term);
}

struct TermCreatorTranslator {
    static unsigned hash(const TermCreatorInput& input)
    {
        return computeHash(input);
    }

    static inline bool equal(const Term* term, const TermCreatorInput& input)
    {
        return *term == input.term;
    }

    static void translate(const Term*& location, const TermCreatorInput& input, unsigned)
    {
        std::unique_ptr<Term> newUniqueTerm(new Term(input.term));
        location = newUniqueTerm.get();
        input.internedTermsStorage.append(WTFMove(newUniqueTerm));
    }
};

const Term* CombinedFiltersAlphabet::interned(const Term& term)
{
    TermCreatorInput input { term, m_internedTermsStorage };
    auto addResult = m_uniqueTerms.add<TermCreatorTranslator>(input);
    return *addResult.iterator;
}

#if CONTENT_EXTENSIONS_PERFORMANCE_REPORTING
size_t CombinedFiltersAlphabet::memoryUsed() const
{
    size_t termsSize = 0;
    for (const auto& termPointer : m_internedTermsStorage)
        termsSize += termPointer->memoryUsed();
    return sizeof(CombinedFiltersAlphabet)
        + termsSize
        + m_uniqueTerms.capacity() * sizeof(Term*)
        + m_internedTermsStorage.capacity() * sizeof(std::unique_ptr<Term>);
}
#endif

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
