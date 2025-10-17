/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

#include <wtf/URL.h>
#include "UserContentTypes.h"
#include "UserScriptTypes.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class UserScript {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(UserScript, WEBCORE_EXPORT);
public:
    ~UserScript() = default;
    UserScript(const UserScript&) = default;
    UserScript(UserScript&&) = default;
    UserScript& operator=(const UserScript&) = default;
    UserScript& operator=(UserScript&&) = default;

    WEBCORE_EXPORT UserScript(String&& source, URL&& = { }, Vector<String>&& allowlist = { }, Vector<String>&& blocklist = { }, UserScriptInjectionTime = UserScriptInjectionTime::DocumentStart, UserContentInjectedFrames = UserContentInjectedFrames::InjectInAllFrames, UserContentMatchParentFrame = UserContentMatchParentFrame::Never);

    const String& source() const { return m_source; }
    const URL& url() const { return m_url; }
    const Vector<String>& allowlist() const { return m_allowlist; }
    const Vector<String>& blocklist() const { return m_blocklist; }
    UserScriptInjectionTime injectionTime() const { return m_injectionTime; }
    UserContentInjectedFrames injectedFrames() const { return m_injectedFrames; }
    UserContentMatchParentFrame matchParentFrame() const { return m_matchParentFrame; }

private:
    String m_source;
    URL m_url;
    Vector<String> m_allowlist;
    Vector<String> m_blocklist;
    UserScriptInjectionTime m_injectionTime { UserScriptInjectionTime::DocumentStart };
    UserContentInjectedFrames m_injectedFrames { UserContentInjectedFrames::InjectInAllFrames };
    UserContentMatchParentFrame m_matchParentFrame { UserContentMatchParentFrame::Never };
};

} // namespace WebCore
