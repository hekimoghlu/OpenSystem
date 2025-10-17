/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "WebGrammarDetail.h"

#include "APIArray.h"
#include "APIString.h"
#include "WKGrammarDetail.h"

namespace WebKit {

Ref<WebGrammarDetail> WebGrammarDetail::create(int location, int length, API::Array* guesses, const String& userDescription)
{
    return adoptRef(*new WebGrammarDetail(location, length, guesses, userDescription));
}

Ref<WebGrammarDetail> WebGrammarDetail::create(const WebCore::GrammarDetail& grammarDetail)
{
    return adoptRef(*new WebGrammarDetail(grammarDetail));
}

WebGrammarDetail::WebGrammarDetail(int location, int length, API::Array* guesses, const String& userDescription)
{
    m_grammarDetail.range = WebCore::CharacterRange(location, length);

    size_t numGuesses = guesses->size();
    m_grammarDetail.guesses.reserveCapacity(numGuesses);
    for (size_t i = 0; i < numGuesses; ++i)
        m_grammarDetail.guesses.append(guesses->at<API::String>(i)->string());

    m_grammarDetail.userDescription = userDescription;
}

Ref<API::Array> WebGrammarDetail::guesses() const
{
    size_t numGuesses = m_grammarDetail.guesses.size();
    Vector<RefPtr<API::Object> > wkGuesses(numGuesses);
    for (unsigned i = 0; i < numGuesses; ++i)
        wkGuesses[i] = API::String::create(m_grammarDetail.guesses[i]);
    return API::Array::create(WTFMove(wkGuesses));
}

WebGrammarDetail::WebGrammarDetail(const WebCore::GrammarDetail& grammarDetail)
    : m_grammarDetail(grammarDetail)
{
}

} // namespace WebKit
