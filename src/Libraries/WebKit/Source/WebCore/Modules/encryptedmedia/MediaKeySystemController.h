/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "MediaKeySystemClient.h"
#include "Supplementable.h"
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaKeySystemRequest;

class MediaKeySystemController : public Supplement<Page> {
    WTF_MAKE_TZONE_ALLOCATED(MediaKeySystemController);
    WTF_MAKE_NONCOPYABLE(MediaKeySystemController);
public:
    explicit MediaKeySystemController(MediaKeySystemClient&);
    ~MediaKeySystemController();

    void requestMediaKeySystem(MediaKeySystemRequest&);
    void cancelMediaKeySystemRequest(MediaKeySystemRequest&);

    void logRequestMediaKeySystemDenial(Document&);

    WEBCORE_EXPORT static ASCIILiteral supplementName();
    static MediaKeySystemController* from(Page*);

private:
    WeakPtr<MediaKeySystemClient> m_client;
};

inline void MediaKeySystemController::requestMediaKeySystem(MediaKeySystemRequest& request)
{
    if (m_client)
        m_client->requestMediaKeySystem(request);
}

inline void MediaKeySystemController::cancelMediaKeySystemRequest(MediaKeySystemRequest& request)
{
    if (m_client)
        m_client->cancelMediaKeySystemRequest(request);
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
