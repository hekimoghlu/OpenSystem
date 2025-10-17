/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include "BackgroundFetchUpdateUIEvent.h"

#include "NotImplemented.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(BackgroundFetchUpdateUIEvent);

Ref<BackgroundFetchUpdateUIEvent> BackgroundFetchUpdateUIEvent::create(const AtomString& type, Init&& init, IsTrusted isTrusted)
{
    auto registration = init.registration;
    return adoptRef(*new BackgroundFetchUpdateUIEvent(type, WTFMove(init), WTFMove(registration), isTrusted));
}

BackgroundFetchUpdateUIEvent::BackgroundFetchUpdateUIEvent(const AtomString& type, ExtendableEventInit&& eventInit, RefPtr<BackgroundFetchRegistration>&& registration, IsTrusted isTrusted)
    : BackgroundFetchEvent(EventInterfaceType::BackgroundFetchUpdateUIEvent, type, WTFMove(eventInit), WTFMove(registration), isTrusted)
{
}

BackgroundFetchUpdateUIEvent::~BackgroundFetchUpdateUIEvent()
{
}

void BackgroundFetchUpdateUIEvent::updateUI(BackgroundFetchUIOptions&&, DOMPromiseDeferred<void>&&)
{
    notImplemented();
}

} // namespace WebCore
