/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "UserScript.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(UserScript);

static WTF::URL generateUserScriptUniqueURL()
{
    static uint64_t identifier;
    return { { }, makeString("user-script:"_s, ++identifier) };
}

UserScript::UserScript(String&& source, URL&& url, Vector<String>&& allowlist, Vector<String>&& blocklist, UserScriptInjectionTime injectionTime, UserContentInjectedFrames injectedFrames, UserContentMatchParentFrame matchParentFrame)
    : m_source(WTFMove(source))
    , m_url(url.isEmpty() ? generateUserScriptUniqueURL() : WTFMove(url))
    , m_allowlist(WTFMove(allowlist))
    , m_blocklist(WTFMove(blocklist))
    , m_injectionTime(injectionTime)
    , m_injectedFrames(injectedFrames)
    , m_matchParentFrame(matchParentFrame)
{
}

} // namespace WebCore
