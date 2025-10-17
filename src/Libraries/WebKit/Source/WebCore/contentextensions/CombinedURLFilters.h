/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "CombinedFiltersAlphabet.h"
#include "ContentExtensionsDebugging.h"
#include "NFA.h"
#include <wtf/Function.h>
#include <wtf/Forward.h>

namespace WebCore {

namespace ContentExtensions {

struct PrefixTreeVertex;

class WEBCORE_EXPORT CombinedURLFilters {
public:
    CombinedURLFilters();
    ~CombinedURLFilters();

    void addPattern(uint64_t actionId, const Vector<Term>& pattern);
    bool processNFAs(size_t maxNFASize, Function<bool(NFA&&)>&&);
    bool isEmpty() const;

#if CONTENT_EXTENSIONS_PERFORMANCE_REPORTING
    size_t memoryUsed() const;
#endif
#if CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING
    void print() const;
#endif
    
private:
    CombinedFiltersAlphabet m_alphabet;
    std::unique_ptr<PrefixTreeVertex> m_prefixTreeRoot;
    UncheckedKeyHashMap<const PrefixTreeVertex*, ActionList> m_actions;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
