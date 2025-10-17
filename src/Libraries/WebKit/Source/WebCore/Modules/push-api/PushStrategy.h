/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#if ENABLE(DECLARATIVE_WEB_PUSH)

#include "ExceptionOr.h"
#include "PushPermissionState.h"
#include "PushSubscriptionData.h"
#include "PushSubscriptionIdentifier.h"

namespace WebCore {

class WEBCORE_EXPORT PushStrategy {
public:
    virtual ~PushStrategy() = default;

    using SubscribeToPushServiceCallback = CompletionHandler<void(ExceptionOr<PushSubscriptionData>&&)>;
    virtual void windowSubscribeToPushService(const URL& scope, const Vector<uint8_t>& applicationServerKey, SubscribeToPushServiceCallback&&) = 0;

    using UnsubscribeFromPushServiceCallback = CompletionHandler<void(ExceptionOr<bool>&&)>;
    virtual void windowUnsubscribeFromPushService(const URL& scope, std::optional<PushSubscriptionIdentifier>, UnsubscribeFromPushServiceCallback&&) = 0;

    using GetPushSubscriptionCallback = CompletionHandler<void(ExceptionOr<std::optional<PushSubscriptionData>>&&)>;
    virtual void windowGetPushSubscription(const URL& scope, GetPushSubscriptionCallback&&) = 0;

    using GetPushPermissionStateCallback = CompletionHandler<void(ExceptionOr<PushPermissionState>&&)>;
    virtual void windowGetPushPermissionState(const URL& scope, GetPushPermissionStateCallback&&) = 0;
};

} // namespace WebCore
#endif // ENABLE(DECLARATIVE_WEB_PUSH)
