/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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
#include "SourceProvider.h"

namespace JSC {

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StringSourceProvider);

SourceProvider::SourceProvider(const SourceOrigin& sourceOrigin, String&& sourceURL, String&& preRedirectURL, SourceTaintedOrigin taintedness, const TextPosition& startPosition, SourceProviderSourceType sourceType)
    : m_sourceType(sourceType)
    , m_sourceOrigin(sourceOrigin)
    , m_sourceURL(WTFMove(sourceURL))
    , m_preRedirectURL(WTFMove(preRedirectURL))
    , m_startPosition(startPosition)
    , m_taintedness(taintedness)
{
}

SourceProvider::~SourceProvider() = default;

void SourceProvider::getID()
{
    if (!m_id) {
        static std::atomic<SourceID> nextProviderID = nullID;
        m_id = ++nextProviderID;
        RELEASE_ASSERT(m_id);
    }
}

const String& SourceProvider::sourceURLStripped()
{
    if (UNLIKELY(m_sourceURL.isNull()))
        return m_sourceURLStripped;
    if (LIKELY(!m_sourceURLStripped.isNull()))
        return m_sourceURLStripped;
    m_sourceURLStripped = URL(m_sourceURL).strippedForUseAsReport();
    return m_sourceURLStripped;
}

#if ENABLE(WEBASSEMBLY)
BaseWebAssemblySourceProvider::BaseWebAssemblySourceProvider(const SourceOrigin& sourceOrigin, String&& sourceURL)
    : SourceProvider(sourceOrigin, WTFMove(sourceURL), String(), SourceTaintedOrigin::Untainted, TextPosition(), SourceProviderSourceType::WebAssembly)
{
}
#endif

} // namespace JSC

