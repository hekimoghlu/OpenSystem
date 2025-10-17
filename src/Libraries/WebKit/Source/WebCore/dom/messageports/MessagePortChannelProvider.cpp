/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#include "MessagePortChannelProvider.h"

#include "Document.h"
#include "MessagePortChannelProviderImpl.h"
#include "WorkerGlobalScope.h"
#include "WorkletGlobalScope.h"
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static WeakPtr<MessagePortChannelProvider>& globalProvider()
{
    static MainThreadNeverDestroyed<WeakPtr<MessagePortChannelProvider>> globalProvider;
    return globalProvider;
}

MessagePortChannelProvider& MessagePortChannelProvider::singleton()
{
    ASSERT(isMainThread());
    auto& globalProvider = WebCore::globalProvider();
    if (!globalProvider)
        globalProvider = new MessagePortChannelProviderImpl;
    return *globalProvider;
}

void MessagePortChannelProvider::setSharedProvider(MessagePortChannelProvider& provider)
{
    RELEASE_ASSERT(isMainThread());
    auto& globalProvider = WebCore::globalProvider();
    RELEASE_ASSERT(!globalProvider);
    globalProvider = provider;
}

MessagePortChannelProvider& MessagePortChannelProvider::fromContext(ScriptExecutionContext& context)
{
    if (auto document = dynamicDowncast<Document>(context))
        return document->messagePortChannelProvider();

    if (auto workletScope = dynamicDowncast<WorkletGlobalScope>(context))
        return workletScope->messagePortChannelProvider();

    return downcast<WorkerGlobalScope>(context).messagePortChannelProvider();
}

} // namespace WebCore
