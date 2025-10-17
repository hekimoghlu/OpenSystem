/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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

#include "EpochTimeStamp.h"
#include "ExceptionOr.h"

#include <JavaScriptCore/ArrayBuffer.h>
#include <optional>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class PushSubscriptionOptions : public RefCounted<PushSubscriptionOptions> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PushSubscriptionOptions);
public:
    template<typename... Args> static Ref<PushSubscriptionOptions> create(Args&&... args) { return adoptRef(*new PushSubscriptionOptions(std::forward<Args>(args)...)); }
    ~PushSubscriptionOptions();

    bool userVisibleOnly() const;
    const Vector<uint8_t>& serverVAPIDPublicKey() const;
    ExceptionOr<RefPtr<JSC::ArrayBuffer>> applicationServerKey() const;

private:
    explicit PushSubscriptionOptions(Vector<uint8_t>&&applicationServerKey);

    Vector<uint8_t> m_serverVAPIDPublicKey;
    mutable RefPtr<JSC::ArrayBuffer> m_applicationServerKey;
};

} // namespace WebCore
