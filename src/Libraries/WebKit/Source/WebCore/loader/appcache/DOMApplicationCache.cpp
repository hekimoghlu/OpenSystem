/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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
#include "DOMApplicationCache.h"

#include "ApplicationCacheHost.h"
#include "Document.h"
#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMApplicationCache);

DOMApplicationCache::DOMApplicationCache(LocalDOMWindow& window)
    : LocalDOMWindowProperty(&window)
{
    if (auto* host = applicationCacheHost())
        host->setDOMApplicationCache(this);
}

ApplicationCacheHost* DOMApplicationCache::applicationCacheHost() const
{
    auto* frame = this->frame();
    if (!frame)
        return nullptr;
    auto* documentLoader = frame->loader().documentLoader();
    if (!documentLoader)
        return nullptr;
    return &documentLoader->applicationCacheHost();
}

unsigned short DOMApplicationCache::status() const
{
    auto* host = applicationCacheHost();
    if (!host)
        return ApplicationCacheHost::UNCACHED;
    return host->status();
}

ExceptionOr<void> DOMApplicationCache::update()
{
    auto* host = applicationCacheHost();
    if (!host || !host->update())
        return Exception { ExceptionCode::InvalidStateError };
    return { };
}

ExceptionOr<void> DOMApplicationCache::swapCache()
{
    auto* host = applicationCacheHost();
    if (!host || !host->swapCache())
        return Exception { ExceptionCode::InvalidStateError };
    return { };
}

void DOMApplicationCache::abort()
{
    if (auto* host = applicationCacheHost())
        host->abort();
}

ScriptExecutionContext* DOMApplicationCache::scriptExecutionContext() const
{
    auto* window = this->window();
    if (!window)
        return nullptr;
    return window->document();
}

} // namespace WebCore
