/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 12, 2022.
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
#include "UserGestureEmulationScope.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "Document.h"
#include "Page.h"
#include "UserGestureIndicator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(UserGestureEmulationScope);

UserGestureEmulationScope::UserGestureEmulationScope(Page& inspectedPage, bool emulateUserGesture, Document* document)
    : m_pageChromeClient(inspectedPage.chrome().client())
    , m_gestureIndicator(emulateUserGesture ? std::optional<IsProcessingUserGesture>(IsProcessingUserGesture::Yes) : std::nullopt, document)
    , m_emulateUserGesture(emulateUserGesture)
    , m_userWasInteracting(false)
{
    if (m_emulateUserGesture) {
        m_userWasInteracting = m_pageChromeClient.userIsInteracting();
        if (!m_userWasInteracting)
            m_pageChromeClient.setUserIsInteracting(true);
    }
}

UserGestureEmulationScope::~UserGestureEmulationScope()
{
    if (m_emulateUserGesture && !m_userWasInteracting && m_pageChromeClient.userIsInteracting())
        m_pageChromeClient.setUserIsInteracting(false);
}

} // namespace WebCore
