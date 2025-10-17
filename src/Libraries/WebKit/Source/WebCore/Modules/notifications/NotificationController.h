/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#if ENABLE(NOTIFICATIONS)

#include "Page.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class NotificationClient;

class NotificationController : public Supplement<Page> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(NotificationController, WEBCORE_EXPORT);
public:
    explicit NotificationController(NotificationClient*);
    ~NotificationController();

    WEBCORE_EXPORT static ASCIILiteral supplementName();
    static NotificationController* from(Page* page) { return static_cast<NotificationController*>(Supplement<Page>::from(page, supplementName())); }
    WEBCORE_EXPORT static NotificationClient* clientFrom(Page&);

    NotificationClient& client() { return m_client; }

private:
    NotificationClient& m_client;
};

} // namespace WebCore

#endif // ENABLE(NOTIFICATIONS)
