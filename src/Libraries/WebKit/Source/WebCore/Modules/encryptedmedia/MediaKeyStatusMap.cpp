/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "MediaKeyStatusMap.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "JSMediaKeyStatusMap.h"
#include "MediaKeySession.h"
#include "SharedBuffer.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {

MediaKeyStatusMap::MediaKeyStatusMap(const MediaKeySession& session)
    : m_session(&session)
{
}

MediaKeyStatusMap::~MediaKeyStatusMap() = default;

void MediaKeyStatusMap::detachSession()
{
    m_session = nullptr;
}

unsigned long MediaKeyStatusMap::size()
{
    if (!m_session)
        return 0;
    return m_session->statuses().size();
}

static bool keyIdsMatch(const SharedBuffer& a, const BufferSource& b)
{
    if (a.isEmpty())
        return false;
    return equalSpans(a.span(), b.span());
}

bool MediaKeyStatusMap::has(const BufferSource& keyId)
{
    if (!m_session)
        return false;

    auto& statuses = m_session->statuses();
    return std::any_of(statuses.begin(), statuses.end(),
        [&keyId] (auto& it) { return keyIdsMatch(it.first, keyId); });
}

JSC::JSValue MediaKeyStatusMap::get(JSC::JSGlobalObject& state, const BufferSource& keyId)
{
    if (!m_session)
        return JSC::jsUndefined();

    auto& statuses = m_session->statuses();
    auto it = std::find_if(statuses.begin(), statuses.end(),
        [&keyId] (auto& it) { return keyIdsMatch(it.first, keyId); });

    if (it == statuses.end())
        return JSC::jsUndefined();
    return convertEnumerationToJS(state.vm(), it->second);
}

MediaKeyStatusMap::Iterator::Iterator(MediaKeyStatusMap& map)
    : m_map(map)
{
}

std::optional<KeyValuePair<BufferSource::VariantType, MediaKeyStatus>> MediaKeyStatusMap::Iterator::next()
{
    if (!m_map->m_session)
        return std::nullopt;

    auto& statuses = m_map->m_session->statuses();
    if (m_index >= statuses.size())
        return std::nullopt;

    auto& pair = statuses[m_index++];
    auto buffer = ArrayBuffer::create(pair.first->makeContiguous()->span());
    return KeyValuePair<BufferSource::VariantType, MediaKeyStatus> { RefPtr<ArrayBuffer>(WTFMove(buffer)), pair.second };
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
