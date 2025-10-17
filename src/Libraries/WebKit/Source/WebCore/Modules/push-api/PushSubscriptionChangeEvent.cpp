/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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
#include "PushSubscriptionChangeEvent.h"

#include "PushSubscription.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PushSubscriptionChangeEvent);

Ref<PushSubscriptionChangeEvent> PushSubscriptionChangeEvent::create(const AtomString& type, PushSubscriptionChangeEventInit&& initializer, IsTrusted isTrusted)
{
    auto newSubscription = initializer.newSubscription;
    auto oldSubscription = initializer.oldSubscription;
    return create(type, WTFMove(initializer), WTFMove(newSubscription), WTFMove(oldSubscription), isTrusted);
}

Ref<PushSubscriptionChangeEvent> PushSubscriptionChangeEvent::create(const AtomString& type, ExtendableEventInit&& initializer, RefPtr<PushSubscription>&& newSubscription, RefPtr<PushSubscription>&& oldSubscription, IsTrusted isTrusted)
{
    return adoptRef(*new PushSubscriptionChangeEvent(type, WTFMove(initializer), WTFMove(newSubscription), WTFMove(oldSubscription), isTrusted));
}

PushSubscriptionChangeEvent::PushSubscriptionChangeEvent(const AtomString& type, ExtendableEventInit&& eventInit, RefPtr<PushSubscription>&& newSubscription, RefPtr<PushSubscription>&& oldSubscription, IsTrusted isTrusted)
    : ExtendableEvent(EventInterfaceType::PushSubscriptionChangeEvent, type, WTFMove(eventInit), isTrusted)
    , m_newSubscription(WTFMove(newSubscription))
    , m_oldSubscription(WTFMove(oldSubscription))
{
}

PushSubscriptionChangeEvent::~PushSubscriptionChangeEvent() = default;

} // namespace WebCore
