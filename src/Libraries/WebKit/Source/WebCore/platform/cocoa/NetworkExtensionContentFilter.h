/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

#if ENABLE(CONTENT_FILTERING)

#include "PlatformContentFilter.h"
#include <objc/NSObjCRuntime.h>
#include <wtf/Compiler.h>
#include <wtf/OSObjectPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

enum NEFilterSourceStatus : NSInteger;

OBJC_CLASS NEFilterSource;
OBJC_CLASS NSData;

namespace WebCore {

class NetworkExtensionContentFilter final : public PlatformContentFilter {
    WTF_MAKE_TZONE_ALLOCATED(NetworkExtensionContentFilter);
    friend UniqueRef<NetworkExtensionContentFilter> WTF::makeUniqueRefWithoutFastMallocCheck<NetworkExtensionContentFilter>();

public:
    static UniqueRef<NetworkExtensionContentFilter> create();

    void willSendRequest(ResourceRequest&, const ResourceResponse&) override;
    void responseReceived(const ResourceResponse&) override;
    void addData(const SharedBuffer&) override;
    void finishedAddingData() override;
    Ref<FragmentedSharedBuffer> replacementData() const override;
    ContentFilterUnblockHandler unblockHandler() const override;

    WEBCORE_EXPORT static bool isRequired();

private:
    static bool enabled();

    NetworkExtensionContentFilter() = default;
    void initialize(const URL* = nullptr);
    void handleDecision(NEFilterSourceStatus, NSData *replacementData);

    OSObjectPtr<dispatch_queue_t> m_queue;
    RetainPtr<NSData> m_replacementData;
    RetainPtr<NEFilterSource> m_neFilterSource;
};

} // namespace WebCore

#endif // ENABLE(CONTENT_FILTERING)
