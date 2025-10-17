/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#include "SharedStringHashTableReadOnly.h"
#include "VisitedLinkTableIdentifier.h"
#include <WebCore/SharedMemory.h>
#include <WebCore/VisitedLinkStore.h>

namespace WebKit {

class VisitedLinkTableController final : public WebCore::VisitedLinkStore, public IPC::MessageReceiver {
public:
    static Ref<VisitedLinkTableController> getOrCreate(VisitedLinkTableIdentifier);
    virtual ~VisitedLinkTableController();

    void ref() const final { WebCore::VisitedLinkStore::ref(); }
    void deref() const final { WebCore::VisitedLinkStore::deref(); }

private:
    explicit VisitedLinkTableController(VisitedLinkTableIdentifier);

    // WebCore::VisitedLinkStore.
    bool isLinkVisited(WebCore::Page&, WebCore::SharedStringHash, const URL& baseURL, const AtomString& attributeURL) override;
    void addVisitedLink(WebCore::Page&, WebCore::SharedStringHash) override;

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void setVisitedLinkTable(WebCore::SharedMemory::Handle&&);
    void visitedLinkStateChanged(const Vector<WebCore::SharedStringHash>&);
    void allVisitedLinkStateChanged();
    void removeAllVisitedLinks();

    VisitedLinkTableIdentifier m_identifier;
    SharedStringHashTableReadOnly m_visitedLinkTable;
};

} // namespace WebKit
