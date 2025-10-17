/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 16, 2023.
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

#if ENABLE(UI_SIDE_COMPOSITING)

#include "MessageReceiver.h"
#include "VisibleContentRectUpdateInfo.h"
#include <WebCore/PageIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/Ref.h>

namespace WTF {
class WorkQueue;
}

namespace WebKit {

class WebProcess;

class ViewUpdateDispatcher final: private IPC::MessageReceiver {
public:
    ViewUpdateDispatcher(WebProcess&);
    ~ViewUpdateDispatcher();

    void ref() const final;
    void deref() const final;

    void initializeConnection(IPC::Connection&);

private:
    // IPC::MessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void visibleContentRectUpdate(WebCore::PageIdentifier, const VisibleContentRectUpdateInfo&);

    void dispatchVisibleContentRectUpdate();

    struct UpdateData {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        UpdateData(const VisibleContentRectUpdateInfo& info, MonotonicTime timestamp)
            : visibleContentRectUpdateInfo(info)
            , oldestTimestamp(timestamp) { }

        VisibleContentRectUpdateInfo visibleContentRectUpdateInfo;
        MonotonicTime oldestTimestamp;
    };

    CheckedRef<WebProcess> m_process;
    Ref<WTF::WorkQueue> m_queue;
    Lock m_latestUpdateLock;
    HashMap<WebCore::PageIdentifier, UniqueRef<UpdateData>> m_latestUpdate WTF_GUARDED_BY_LOCK(m_latestUpdateLock);
};

} // namespace WebKit

#endif // ENABLE(UI_SIDE_COMPOSITING)
