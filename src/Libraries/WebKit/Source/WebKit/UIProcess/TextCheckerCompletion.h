/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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

#include "IdentifierTypes.h"
#include <WebCore/TextChecking.h>
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

using SpellDocumentTag = int64_t;

class WebPageProxy;

class TextCheckerCompletion : public RefCounted<TextCheckerCompletion> {
public:
    static Ref<TextCheckerCompletion> create(TextCheckerRequestID, const WebCore::TextCheckingRequestData&, WebPageProxy&);

    const WebCore::TextCheckingRequestData& textCheckingRequestData() const;
    SpellDocumentTag spellDocumentTag();
    void didFinishCheckingText(const Vector<WebCore::TextCheckingResult>&) const;
    void didCancelCheckingText() const;

private:
    TextCheckerCompletion(TextCheckerRequestID, const WebCore::TextCheckingRequestData&, WebPageProxy&);

    Ref<WebPageProxy> protectedPage() const;

    const TextCheckerRequestID m_requestID;
    const WebCore::TextCheckingRequestData m_requestData;
    WeakRef<WebPageProxy> m_page;
};

} // namespace WebKit
