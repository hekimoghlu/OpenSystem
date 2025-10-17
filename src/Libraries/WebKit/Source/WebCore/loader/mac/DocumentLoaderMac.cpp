/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "DocumentLoader.h"

#include "SubresourceLoader.h"

namespace WebCore {

static void scheduleAll(const ResourceLoaderMap& loaders, SchedulePair& pair)
{
    for (RefPtr loader : copyToVector(loaders.values()))
        loader->schedule(pair);
}

static void unscheduleAll(const ResourceLoaderMap& loaders, SchedulePair& pair)
{
    for (RefPtr loader : copyToVector(loaders.values()))
        loader->unschedule(pair);
}

void DocumentLoader::schedule(SchedulePair& pair)
{
    if (RefPtr loader = mainResourceLoader())
        loader->schedule(pair);
    scheduleAll(m_subresourceLoaders, pair);
    scheduleAll(m_plugInStreamLoaders, pair);
    scheduleAll(m_multipartSubresourceLoaders, pair);
}

void DocumentLoader::unschedule(SchedulePair& pair)
{
    if (RefPtr loader = mainResourceLoader())
        loader->unschedule(pair);
    unscheduleAll(m_subresourceLoaders, pair);
    unscheduleAll(m_plugInStreamLoaders, pair);
    unscheduleAll(m_multipartSubresourceLoaders, pair);
}

} // namespace WebCore
