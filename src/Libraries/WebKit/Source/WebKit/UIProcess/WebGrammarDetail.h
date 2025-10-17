/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 2, 2024.
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
#ifndef WebGrammarDetail_h
#define WebGrammarDetail_h

#include "APIObject.h"
#include <WebCore/TextCheckerClient.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>

namespace API {
class Array;
}

namespace WebKit {

class WebGrammarDetail : public API::ObjectImpl<API::Object::Type::GrammarDetail> {
public:
    static Ref<WebGrammarDetail> create(int location, int length, API::Array* guesses, const String& userDescription);
    static Ref<WebGrammarDetail> create(const WebCore::GrammarDetail&);

    int location() const { return m_grammarDetail.range.location; }
    int length() const { return m_grammarDetail.range.length; }
    Ref<API::Array> guesses() const;
    const String& userDescription() const { return m_grammarDetail.userDescription; }

    const WebCore::GrammarDetail& grammarDetail() { return m_grammarDetail; }

private:
    WebGrammarDetail(int location, int length, API::Array* guesses, const String& userDescription);
    explicit WebGrammarDetail(const WebCore::GrammarDetail&);

    WebCore::GrammarDetail m_grammarDetail;
};

} // namespace WebKit

#endif // WebGrammarDetail_h
