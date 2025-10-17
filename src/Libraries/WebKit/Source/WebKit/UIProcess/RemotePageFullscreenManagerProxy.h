/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

#if ENABLE(FULLSCREEN_API)

#include "MessageReceiver.h"
#include <WebCore/PageIdentifier.h>

namespace WebKit {

class WebFullScreenManagerProxy;
class WebProcessProxy;

class RemotePageFullscreenManagerProxy : public IPC::MessageReceiver, public RefCounted<RemotePageFullscreenManagerProxy> {
public:
    static Ref<RemotePageFullscreenManagerProxy> create(WebCore::PageIdentifier, WebFullScreenManagerProxy*, WebProcessProxy&);

    ~RemotePageFullscreenManagerProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    RemotePageFullscreenManagerProxy(WebCore::PageIdentifier, WebFullScreenManagerProxy*, WebProcessProxy&);

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    const WebCore::PageIdentifier m_identifier;
    const WeakPtr<WebFullScreenManagerProxy> m_manager;
    const Ref<WebProcessProxy> m_process;
};

}
#endif
