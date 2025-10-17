/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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

#include "MockContentFilterSettings.h"
#include "PlatformContentFilter.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class MockContentFilter final : public PlatformContentFilter {
    WTF_MAKE_TZONE_ALLOCATED(MockContentFilter);
    friend UniqueRef<MockContentFilter> WTF::makeUniqueRefWithoutFastMallocCheck<MockContentFilter>();

public:
    static void ensureInstalled();
    static UniqueRef<MockContentFilter> create();

    void willSendRequest(ResourceRequest&, const ResourceResponse&) override;
    void responseReceived(const ResourceResponse&) override;
    void addData(const SharedBuffer&) override;
    void finishedAddingData() override;
    Ref<FragmentedSharedBuffer> replacementData() const override;
#if ENABLE(CONTENT_FILTERING)
    ContentFilterUnblockHandler unblockHandler() const override;
#endif
    String unblockRequestDeniedScript() const override;

private:
    static bool enabled();

    MockContentFilter() = default;
    void maybeDetermineStatus(MockContentFilterSettings::DecisionPoint);

    Vector<char> m_replacementData;
};

} // namespace WebCore
