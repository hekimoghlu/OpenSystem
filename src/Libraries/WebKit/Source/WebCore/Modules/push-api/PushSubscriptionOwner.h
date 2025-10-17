/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 28, 2022.
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

#include "JSDOMPromiseDeferredForward.h"
#include "PushSubscription.h"
#include <wtf/AbstractRefCounted.h>

namespace WebCore {

class PushSubscription;

enum class PushPermissionState : uint8_t;

class PushSubscriptionOwner : public AbstractRefCounted {
public:
    virtual ~PushSubscriptionOwner() = default;

    virtual bool isActive() const = 0;

    virtual void subscribeToPushService(const Vector<uint8_t>& applicationServerKey, DOMPromiseDeferred<IDLInterface<PushSubscription>>&&) = 0;
    virtual void unsubscribeFromPushService(std::optional<PushSubscriptionIdentifier>, DOMPromiseDeferred<IDLBoolean>&&) = 0;
    virtual void getPushSubscription(DOMPromiseDeferred<IDLNullable<IDLInterface<PushSubscription>>>&&) = 0;
    virtual void getPushPermissionState(DOMPromiseDeferred<IDLEnumeration<PushPermissionState>>&&) = 0;
};

}
