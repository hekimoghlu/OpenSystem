/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#include "TextCheckerCompletion.h"

#include "WebPageProxy.h"

namespace WebKit {
using namespace WebCore;

Ref<TextCheckerCompletion> TextCheckerCompletion::create(TextCheckerRequestID requestID, const TextCheckingRequestData& requestData, WebPageProxy& page)
{
    return adoptRef(*new TextCheckerCompletion(requestID, requestData, page));
}

TextCheckerCompletion::TextCheckerCompletion(TextCheckerRequestID requestID, const TextCheckingRequestData& requestData, WebPageProxy& page)
    : m_requestID(requestID)
    , m_requestData(requestData)
    , m_page(page)
{
}

Ref<WebPageProxy> TextCheckerCompletion::protectedPage() const
{
    return m_page.get();
}

const TextCheckingRequestData& TextCheckerCompletion::textCheckingRequestData() const
{
    return m_requestData;
}

int64_t TextCheckerCompletion::spellDocumentTag()
{
    return protectedPage()->spellDocumentTag();
}

void TextCheckerCompletion::didFinishCheckingText(const Vector<TextCheckingResult>& result) const
{
    if (result.isEmpty())
        didCancelCheckingText();

    protectedPage()->didFinishCheckingText(m_requestID, result);
}

void TextCheckerCompletion::didCancelCheckingText() const
{
    protectedPage()->didCancelCheckingText(m_requestID);
}

} // namespace WebKit
