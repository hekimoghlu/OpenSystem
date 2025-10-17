/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 7, 2023.
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

#if ENABLE(IPC_TESTING_API)

#include "MessageReceiver.h"
#include <wtf/RefCounted.h>

namespace IPC {
class Connection;
class IPCSemaphore;
}

namespace WebKit {

// Proxy interface to test various IPC stream related activities.
// Currently this is not instantiated. This only exists due to the IPCStreamTesterProxy
// messages that are caught in the JS IPC_TESTING_API tests. The messages need to
// compile the IPCStreamTesterProxyMessageReceiver.cpp, so this class definition is needed.
class IPCStreamTesterProxy final : public IPC::MessageReceiver, public RefCounted<IPCStreamTesterProxy> {
public:
    ~IPCStreamTesterProxy() = default;

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // IPC::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

private:
    IPCStreamTesterProxy() = default;

    // Messages.
    void wasCreated(IPC::Semaphore&&, IPC::Semaphore&&) { }
};

}

#endif
