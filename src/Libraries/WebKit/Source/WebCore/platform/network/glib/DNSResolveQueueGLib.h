/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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

#include "DNSResolveQueue.h"

#if USE(GLIB)

#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/glib/GRefPtr.h>

namespace WebCore {

class DNSResolveQueueGLib final : public DNSResolveQueue {
public:
    DNSResolveQueueGLib() = default;

    void resolve(const String& hostname, uint64_t identifier, DNSCompletionHandler&&) final;
    void stopResolve(uint64_t identifier) final;

private:
    struct Request {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        Request(uint64_t identifier, DNSCompletionHandler&& completionHandler)
            : identifier(identifier)
            , completionHandler(WTFMove(completionHandler))
        {
        }

        uint64_t identifier { 0 };
        DNSCompletionHandler completionHandler;
    };

    void updateIsUsingProxy() final;
    void platformResolve(const String&) final;

    HashMap<uint64_t, GRefPtr<GCancellable>> m_requestCancellables;
};

using DNSResolveQueuePlatform = DNSResolveQueueGLib;

} // namespace WebCore

#endif // USE(GLIB)
