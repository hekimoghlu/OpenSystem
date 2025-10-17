/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#include "WebInspectorInterruptDispatcher.h"

#include "Connection.h"
#include "WebInspectorInterruptDispatcherMessages.h"
#include "WebProcess.h"
#include <JavaScriptCore/VM.h>
#include <WebCore/CommonVM.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

WebInspectorInterruptDispatcher::WebInspectorInterruptDispatcher(WebProcess& process)
    : m_process(process)
    , m_queue(WorkQueue::create("com.apple.WebKit.WebInspectorInterruptDispatcher"_s))
{
}

WebInspectorInterruptDispatcher::~WebInspectorInterruptDispatcher()
{
    ASSERT_NOT_REACHED();
}

void WebInspectorInterruptDispatcher::ref() const
{
    m_process->ref();
}

void WebInspectorInterruptDispatcher::deref() const
{
    m_process->deref();
}

void WebInspectorInterruptDispatcher::initializeConnection(IPC::Connection& connection)
{
    connection.addMessageReceiver(m_queue.get(), *this, Messages::WebInspectorInterruptDispatcher::messageReceiverName());
}

void WebInspectorInterruptDispatcher::notifyNeedDebuggerBreak()
{
    // If the web process has not been fully initialized yet, then there
    // is no VM to be notified and thus no infinite loop to break. Bail out.
    if (!WebCore::commonVMOrNull())
        return;

    Ref vm = WebCore::commonVM();
    vm->notifyNeedDebuggerBreak();
}

} // namespace WebKit
