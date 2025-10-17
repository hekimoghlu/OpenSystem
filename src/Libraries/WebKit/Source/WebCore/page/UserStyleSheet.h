/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 12, 2025.
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

#include "PageIdentifier.h"
#include "UserContentTypes.h"
#include "UserStyleSheetTypes.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>

namespace WebCore {

class UserStyleSheet {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(UserStyleSheet, WEBCORE_EXPORT);
public:
    UserStyleSheet()
        : m_injectedFrames(UserContentInjectedFrames::InjectInAllFrames)
        , m_level(UserStyleLevel::User)
    {
    }

    WEBCORE_EXPORT UserStyleSheet(const String&, const URL&, Vector<String>&& = { }, Vector<String>&& = { }, UserContentInjectedFrames = UserContentInjectedFrames::InjectInAllFrames, UserContentMatchParentFrame = UserContentMatchParentFrame::Never, UserStyleLevel = UserStyleLevel::User, std::optional<PageIdentifier> = std::nullopt);

    const String& source() const { return m_source; }
    const URL& url() const { return m_url; }
    const Vector<String>& allowlist() const { return m_allowlist; }
    const Vector<String>& blocklist() const { return m_blocklist; }
    UserContentInjectedFrames injectedFrames() const { return m_injectedFrames; }
    UserContentMatchParentFrame matchParentFrame() const { return m_matchParentFrame; }
    UserStyleLevel level() const { return m_level; }
    std::optional<PageIdentifier> pageID() const { return m_pageID; }

private:
    String m_source;
    URL m_url;
    Vector<String> m_allowlist;
    Vector<String> m_blocklist;
    UserContentInjectedFrames m_injectedFrames { UserContentInjectedFrames::InjectInAllFrames };
    UserContentMatchParentFrame m_matchParentFrame { UserContentMatchParentFrame::Never };
    UserStyleLevel m_level { UserStyleLevel::User };
    std::optional<PageIdentifier> m_pageID;
};

} // namespace WebCore
