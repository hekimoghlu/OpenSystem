/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#include "MediaKeySystemController.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "Document.h"
#include "HTMLIFrameElement.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "MediaKeySystemRequest.h"
#include "Page.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MediaKeySystemController);

ASCIILiteral MediaKeySystemController::supplementName()
{
    return "MediaKeySystemController"_s;
}

MediaKeySystemController* MediaKeySystemController::from(Page* page)
{
    return static_cast<MediaKeySystemController*>(Supplement<Page>::from(page, MediaKeySystemController::supplementName()));
}

MediaKeySystemController::MediaKeySystemController(MediaKeySystemClient& client)
    : m_client(client)
{
}

MediaKeySystemController::~MediaKeySystemController()
{
    if (m_client)
        m_client->pageDestroyed();
}

void provideMediaKeySystemTo(Page& page, MediaKeySystemClient& client)
{
    Supplement<Page>::provideTo(&page, MediaKeySystemController::supplementName(), makeUnique<MediaKeySystemController>(client));
}

void MediaKeySystemController::logRequestMediaKeySystemDenial(Document& document)
{
    if (RefPtr window = document.domWindow())
        window->printErrorMessage("Not allowed to access MediaKeySystem."_str);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM)
