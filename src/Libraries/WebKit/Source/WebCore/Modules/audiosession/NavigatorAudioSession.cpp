/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 24, 2024.
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
#include "NavigatorAudioSession.h"

#if ENABLE(DOM_AUDIO_SESSION)

#include "DOMAudioSession.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(NavigatorAudioSession);

NavigatorAudioSession::NavigatorAudioSession() = default;

NavigatorAudioSession::~NavigatorAudioSession() = default;

RefPtr<DOMAudioSession> NavigatorAudioSession::audioSession(Navigator& navigator)
{
    auto* navigatorAudioSession = NavigatorAudioSession::from(navigator);
    if (!navigatorAudioSession->m_audioSession)
        navigatorAudioSession->m_audioSession = DOMAudioSession::create(navigator.scriptExecutionContext());
    return navigatorAudioSession->m_audioSession;
}

NavigatorAudioSession* NavigatorAudioSession::from(Navigator& navigator)
{
    auto* supplement = static_cast<NavigatorAudioSession*>(Supplement<Navigator>::from(&navigator, supplementName()));
    if (!supplement) {
        auto newSupplement = makeUnique<NavigatorAudioSession>();
        supplement = newSupplement.get();
        provideTo(&navigator, supplementName(), WTFMove(newSupplement));
    }
    return supplement;
}

ASCIILiteral NavigatorAudioSession::supplementName()
{
    return "NavigatorAudioSession"_s;
}

}

#endif // ENABLE(DOM_AUDIO_SESSION)
