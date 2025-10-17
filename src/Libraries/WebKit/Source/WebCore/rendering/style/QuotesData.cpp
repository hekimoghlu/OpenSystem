/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#include "QuotesData.h"
#include <wtf/StdLibExtras.h>
#include <wtf/ZippedRange.h>

namespace WebCore {

static size_t sizeForQuotesDataWithQuoteCount(unsigned count)
{
    return sizeof(QuotesData) + sizeof(std::pair<String, String>) * count;
}

Ref<QuotesData> QuotesData::create(const Vector<std::pair<String, String>>& quotes)
{
    void* slot = fastMalloc(sizeForQuotesDataWithQuoteCount(quotes.size()));
    return adoptRef(*new (NotNull, slot) QuotesData(quotes));
}

QuotesData::QuotesData(const Vector<std::pair<String, String>>& quotes)
    : m_quoteCount(quotes.size())
{
    for (auto [quotePair, quote] : zippedRange(quotePairs(), quotes))
        new (NotNull, &quotePair) std::pair<String, String>(quote);
}

QuotesData::~QuotesData()
{
    for (auto& quotePair : quotePairs())
        quotePair.~pair<String, String>();
}

std::span<const std::pair<String, String>> QuotesData::quotePairs() const
{
    return unsafeMakeSpan(m_quotePairs, m_quoteCount);
}

std::span<std::pair<String, String>> QuotesData::quotePairs()
{
    return unsafeMakeSpan(m_quotePairs, m_quoteCount);
}

const String& QuotesData::openQuote(unsigned index) const
{
    auto quotePairs = this->quotePairs();
    if (quotePairs.empty())
        return emptyString();

    if (index < quotePairs.size())
        return quotePairs[index].first;
    return quotePairs.back().first;
}

const String& QuotesData::closeQuote(unsigned index) const
{
    auto quotePairs = this->quotePairs();
    if (quotePairs.empty())
        return emptyString();

    if (index < quotePairs.size())
        return quotePairs[index].second;
    return quotePairs.back().second;
}

bool operator==(const QuotesData& a, const QuotesData& b)
{
    if (a.m_quoteCount != b.m_quoteCount)
        return false;

    for (auto [aPair, bPair] : zippedRange(a.quotePairs(), b.quotePairs())) {
        if (aPair != bPair)
            return false;
    }

    return true;
}

} // namespace WebCore
