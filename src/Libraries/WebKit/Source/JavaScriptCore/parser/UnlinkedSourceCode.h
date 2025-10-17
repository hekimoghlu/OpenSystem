/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 5, 2024.
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

#include "SourceProvider.h"
#include <wtf/RefPtr.h>

namespace JSC {

    class UnlinkedSourceCode {
        template<typename SourceType>
        friend class CachedUnlinkedSourceCodeShape;
        friend class CachedSourceCodeWithoutProvider;

    public:
        UnlinkedSourceCode()
            : m_provider(nullptr)
            , m_startOffset(0)
            , m_endOffset(0)
        {
        }

        UnlinkedSourceCode(WTF::HashTableDeletedValueType)
            : m_provider(WTF::HashTableDeletedValue)
        {
        }

        UnlinkedSourceCode(Ref<SourceProvider>&& provider)
            : m_provider(WTFMove(provider))
            , m_startOffset(0)
            , m_endOffset(m_provider->source().length())
        {
        }

        UnlinkedSourceCode(Ref<SourceProvider>&& provider, int startOffset, int endOffset)
            : m_provider(WTFMove(provider))
            , m_startOffset(startOffset)
            , m_endOffset(endOffset)
        {
        }

        UnlinkedSourceCode(RefPtr<SourceProvider>&& provider, int startOffset, int endOffset)
            : m_provider(WTFMove(provider))
            , m_startOffset(startOffset)
            , m_endOffset(endOffset)
        {
        }

        bool isHashTableDeletedValue() const { return m_provider.isHashTableDeletedValue(); }

        SourceProvider& provider() const
        {
            return *m_provider;
        }

        unsigned hash() const
        {
            ASSERT(m_provider);
            return m_provider->hash();
        }

        StringView view() const
        {
            if (!m_provider)
                return StringView();
            return m_provider->getRange(m_startOffset, m_endOffset);
        }
        
        CString toUTF8() const;
        
        bool isNull() const { return !m_provider; }
        int startOffset() const { return m_startOffset; }
        int endOffset() const { return m_endOffset; }
        int length() const { return m_endOffset - m_startOffset; }

        friend bool operator==(const UnlinkedSourceCode&, const UnlinkedSourceCode&) = default;

    protected:
        // FIXME: Make it Ref<SourceProvidier>.
        // https://bugs.webkit.org/show_bug.cgi?id=168325
        RefPtr<SourceProvider> m_provider;
        int m_startOffset;
        int m_endOffset;
    };

} // namespace JSC
