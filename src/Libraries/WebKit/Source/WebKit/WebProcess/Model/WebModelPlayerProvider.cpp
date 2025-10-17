/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#include "WebModelPlayerProvider.h"

#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/ModelPlayer.h>
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)
#include "ARKitInlinePreviewModelPlayerMac.h"
#endif

#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
#include "ARKitInlinePreviewModelPlayerIOS.h"
#endif

#if HAVE(SCENEKIT)
#include <WebCore/SceneKitModelPlayer.h>
#endif

#if ENABLE(MODEL_PROCESS)
#include "ModelProcessModelPlayer.h"
#include "ModelProcessModelPlayerManager.h"
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebModelPlayerProvider);

WebModelPlayerProvider::WebModelPlayerProvider(WebPage& page)
    : m_page { page }
{
}

WebModelPlayerProvider::~WebModelPlayerProvider() = default;

// MARK: - WebCore::ModelPlayerProvider overrides.

RefPtr<WebCore::ModelPlayer> WebModelPlayerProvider::createModelPlayer(WebCore::ModelPlayerClient& client)
{
    Ref page = m_page.get();
    UNUSED_PARAM(page);
#if ENABLE(MODEL_PROCESS)
    if (page->corePage()->settings().modelProcessEnabled())
        return WebProcess::singleton().modelProcessModelPlayerManager().createModelProcessModelPlayer(page, client);
#endif
#if ENABLE(ARKIT_INLINE_PREVIEW_MAC)
    if (page->useARKitForModel())
        return ARKitInlinePreviewModelPlayerMac::create(page, client);
#endif
#if HAVE(SCENEKIT)
    if (page->useSceneKitForModel())
        return WebCore::SceneKitModelPlayer::create(client);
#endif
#if ENABLE(ARKIT_INLINE_PREVIEW_IOS)
    return ARKitInlinePreviewModelPlayerIOS::create(page, client);
#endif

    UNUSED_PARAM(client);
    return nullptr;
}

void WebModelPlayerProvider::deleteModelPlayer(WebCore::ModelPlayer& modelPlayer)
{
#if ENABLE(MODEL_PROCESS)
    Ref page = m_page.get();
    if (page->corePage()->settings().modelProcessEnabled())
        return WebProcess::singleton().modelProcessModelPlayerManager().deleteModelProcessModelPlayer(modelPlayer);
#else
    UNUSED_PARAM(modelPlayer);
#endif
}

}
