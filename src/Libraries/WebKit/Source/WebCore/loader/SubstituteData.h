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

#include "ResourceResponse.h"
#include "SharedBuffer.h"
#include <wtf/URL.h>

namespace WebCore {

enum class SessionHistoryVisibility : bool { Visible, Hidden };

class SubstituteData {
public:
    using SessionHistoryVisibility = WebCore::SessionHistoryVisibility;

    SubstituteData() = default;

    SubstituteData(RefPtr<FragmentedSharedBuffer>&& content, const URL& failingURL, const ResourceResponse& response, SessionHistoryVisibility shouldRevealToSessionHistory)
        : m_content(WTFMove(content))
        , m_failingURL(failingURL)
        , m_response(response)
        , m_shouldRevealToSessionHistory(shouldRevealToSessionHistory)
    {
    }

    bool isValid() const { return m_content != nullptr; }
    SessionHistoryVisibility shouldRevealToSessionHistory() const { return m_shouldRevealToSessionHistory; }

    const RefPtr<FragmentedSharedBuffer>& content() const { return m_content; }
    const String& mimeType() const { return m_response.mimeType(); }
    const String& textEncoding() const { return m_response.textEncodingName(); }
    const URL& failingURL() const { return m_failingURL; }
    const ResourceResponse& response() const { return m_response; }

private:
    RefPtr<FragmentedSharedBuffer> m_content;
    URL m_failingURL;
    ResourceResponse m_response;
    SessionHistoryVisibility m_shouldRevealToSessionHistory { SessionHistoryVisibility::Hidden };
};

} // namespace WebCore
