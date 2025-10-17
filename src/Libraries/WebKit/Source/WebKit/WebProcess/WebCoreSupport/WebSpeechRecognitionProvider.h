/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#include "WebPage.h"
#include "WebSpeechRecognitionConnection.h"
#include <WebCore/SpeechRecognitionProvider.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

class WebSpeechRecognitionProvider final : public WebCore::SpeechRecognitionProvider {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebSpeechRecognitionProvider);
public:
    explicit WebSpeechRecognitionProvider(WebCore::PageIdentifier identifier)
        : m_pageIdentifier(identifier)
    {
    }

    WebCore::SpeechRecognitionConnection& speechRecognitionConnection() final
    {
        if (!m_connection)
            m_connection = WebSpeechRecognitionConnection::create(m_pageIdentifier);

        return *m_connection;
    }

private:
    WebCore::PageIdentifier m_pageIdentifier;
    RefPtr<WebSpeechRecognitionConnection> m_connection;
};

} // namespace WebKit
