/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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
#include "MediaDeviceSandboxExtensions.h"

#if ENABLE(MEDIA_STREAM)


namespace WebKit {

MediaDeviceSandboxExtensions::MediaDeviceSandboxExtensions(Vector<String> ids, Vector<SandboxExtension::Handle>&& handles, SandboxExtension::Handle&& machBootstrapHandle)
    : m_ids(ids)
    , m_handles(WTFMove(handles))
    , m_machBootstrapHandle(WTFMove(machBootstrapHandle))
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(m_ids.size() == m_handles.size());
}

std::pair<String, RefPtr<SandboxExtension>> MediaDeviceSandboxExtensions::operator[](size_t i)
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(m_ids.size() == m_handles.size());
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(i < m_ids.size());
    return { m_ids[i], SandboxExtension::create(WTFMove(m_handles[i])) };
}

size_t MediaDeviceSandboxExtensions::size() const
{
    return m_ids.size();
}

} // namespace WebKit

#endif // ENABLE(MEDIA_STREAM)
