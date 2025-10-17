/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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

#include "MessageReceiver.h"
#include <wtf/CheckedRef.h>
#include <wtf/Ref.h>

namespace WTF {
class WorkQueue;
}

namespace WebKit {

class WebProcess;

class WebInspectorInterruptDispatcher final : private IPC::MessageReceiver {
public:
    explicit WebInspectorInterruptDispatcher(WebProcess&);
    ~WebInspectorInterruptDispatcher();
    
    void initializeConnection(IPC::Connection&);

    void ref() const final;
    void deref() const final;
    
private:
    // IPC::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    
    void notifyNeedDebuggerBreak();
    
    CheckedRef<WebProcess> m_process;
    Ref<WTF::WorkQueue> m_queue;
};

} // namespace WebKit
