/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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

#include "ActiveDOMCallback.h"
#include "CallbackResult.h"
#include "ContextDestructionObserverInlines.h"
#include "RTCPeerConnection.h"

namespace WebCore {

class RTCLogsCallback : public RefCounted<RTCLogsCallback>, public ActiveDOMCallback {
public:
    using ActiveDOMCallback::ActiveDOMCallback;

    struct Logs {
        String type;
        String message;
        String level;
        RefPtr<RTCPeerConnection> connection;
    };

    virtual CallbackResult<void> handleEvent(const Logs&) = 0;
    virtual CallbackResult<void> handleEventRethrowingException(const Logs&) = 0;

private:
    virtual bool hasCallback() const = 0;
};

} // namespace WebCore
