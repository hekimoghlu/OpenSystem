/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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

#include "CachedTextTrack.h"

#include "CachedResourceClient.h"
#include "CachedResourceClientWalker.h"
#include "CachedResourceLoader.h"
#include "SharedBuffer.h"
#include "TextResourceDecoder.h"

#if ENABLE(VIDEO)

namespace WebCore {

CachedTextTrack::CachedTextTrack(CachedResourceRequest&& request, PAL::SessionID sessionID, const CookieJar* cookieJar)
    : CachedResource(WTFMove(request), Type::TextTrackResource, sessionID, cookieJar)
{
}

void CachedTextTrack::doUpdateBuffer(const FragmentedSharedBuffer* data)
{
    ASSERT(dataBufferingPolicy() == DataBufferingPolicy::BufferData);
    m_data = data ? data->makeContiguous() : RefPtr<SharedBuffer>();
    setEncodedSize(data ? data->size() : 0);

    CachedResourceClientWalker<CachedResourceClient> walker(*this);
    while (CachedResourceClient* client = walker.next())
        client->deprecatedDidReceiveCachedResource(*this);
}

void CachedTextTrack::updateBuffer(const FragmentedSharedBuffer& data)
{
    doUpdateBuffer(&data);
    CachedResource::updateBuffer(data);
}

void CachedTextTrack::finishLoading(const FragmentedSharedBuffer* data, const NetworkLoadMetrics& metrics)
{
    doUpdateBuffer(data);
    CachedResource::finishLoading(data, metrics);
}

}

#endif // ENABLE(VIDEO)
