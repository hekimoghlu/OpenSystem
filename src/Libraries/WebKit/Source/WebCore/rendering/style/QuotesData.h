/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

#if COMPILER(MSVC)
#pragma warning(push)
#pragma warning(disable: 4200) // Disable "zero-sized array in struct/union" warning
#endif

class QuotesData : public RefCounted<QuotesData> {
public:
    static Ref<QuotesData> create(const Vector<std::pair<String, String>>& quotes);
    ~QuotesData();

    friend bool operator==(const QuotesData&, const QuotesData&);

    const String& openQuote(unsigned index) const;
    const String& closeQuote(unsigned index) const;

    unsigned size() const { return m_quoteCount; }

private:
    explicit QuotesData(const Vector<std::pair<String, String>>& quotes);

    std::span<const std::pair<String, String>> quotePairs() const;
    std::span<std::pair<String, String>> quotePairs();

    unsigned m_quoteCount;
    std::pair<String, String> m_quotePairs[0];
};

#if COMPILER(MSVC)
#pragma warning(pop)
#endif

} // namespace WebCore
