/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#include "APIObject.h"
#include "MessageReceiver.h"
#include "SharedStringHashStore.h"
#include "VisitedLinkTableIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/Identified.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakHashSet.h>

namespace WebKit {

class WebProcessProxy;
    
class VisitedLinkStore final : public API::ObjectImpl<API::Object::Type::VisitedLinkStore>, public IPC::MessageReceiver, public Identified<VisitedLinkTableIdentifier>, private SharedStringHashStore::Client {
public:
    static Ref<VisitedLinkStore> create();
    VisitedLinkStore();

    virtual ~VisitedLinkStore();

    void ref() const final { API::ObjectImpl<API::Object::Type::VisitedLinkStore>::ref(); }
    void deref() const final { API::ObjectImpl<API::Object::Type::VisitedLinkStore>::deref(); }

    void addProcess(WebProcessProxy&);
    void removeProcess(WebProcessProxy&);

    void addVisitedLinkHash(WebCore::SharedStringHash);
    bool containsVisitedLinkHash(WebCore::SharedStringHash);
    void removeVisitedLinkHash(WebCore::SharedStringHash);
    void removeAll();

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // SharedStringHashStore::Client
    void didInvalidateSharedMemory() final;
    void didUpdateSharedStringHashes(const Vector<WebCore::SharedStringHash>& addedHashes, const Vector<WebCore::SharedStringHash>& removedHashes) final;

    void addVisitedLinkHashFromPage(WebPageProxyIdentifier, WebCore::SharedStringHash);

    void sendStoreHandleToProcess(WebProcessProxy&);

    WeakHashSet<WebProcessProxy> m_processes;
    SharedStringHashStore m_linkHashStore;
};

} // namespace WebKit
