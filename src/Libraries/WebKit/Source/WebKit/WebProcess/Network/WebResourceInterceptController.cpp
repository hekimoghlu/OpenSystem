/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#include "WebResourceInterceptController.h"

namespace WebKit {

bool WebResourceInterceptController::isIntercepting(WebCore::ResourceLoaderIdentifier identifier) const
{
    return m_interceptedResponseQueue.contains(identifier);
}

void WebResourceInterceptController::beginInterceptingResponse(WebCore::ResourceLoaderIdentifier identifier)
{
    m_interceptedResponseQueue.set(identifier, Deque<Function<void()>>());
}

void WebResourceInterceptController::continueResponse(WebCore::ResourceLoaderIdentifier identifier)
{
    auto queue = m_interceptedResponseQueue.take(identifier);
    for (auto& callback : queue)
        callback();
}

void WebResourceInterceptController::interceptedResponse(WebCore::ResourceLoaderIdentifier identifier)
{
    m_interceptedResponseQueue.remove(identifier);
}

void WebResourceInterceptController::defer(WebCore::ResourceLoaderIdentifier identifier, Function<void()>&& function)
{
    ASSERT(isIntercepting(identifier));

    auto iterator = m_interceptedResponseQueue.find(identifier);
    if (iterator != m_interceptedResponseQueue.end())
        iterator->value.append(WTFMove(function));
}

} // namespace WebKit
