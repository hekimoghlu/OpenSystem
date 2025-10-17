/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#include "DisplayListReplayer.h"

#include "ControlFactory.h"
#include "DisplayList.h"
#include "Logging.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace DisplayList {

Replayer::Replayer(GraphicsContext& context, const DisplayList& displayList)
    : Replayer(context, displayList.items(), displayList.resourceHeap(), ControlFactory::shared(), displayList.replayOptions())
{
}

Replayer::Replayer(GraphicsContext& context, const Vector<Item>& items, const ResourceHeap& resourceHeap, ControlFactory& controlFactory, OptionSet<ReplayOption> options)
    : m_context(context)
    , m_items(items)
    , m_resourceHeap(resourceHeap)
    , m_controlFactory(controlFactory)
    , m_options(options)
{
}

ReplayResult Replayer::replay(const FloatRect& initialClip, bool trackReplayList)
{
    LOG_WITH_STREAM(DisplayLists, stream << "\nReplaying with clip " << initialClip);
    UNUSED_PARAM(initialClip);

    std::unique_ptr<DisplayList> replayList;
    if (UNLIKELY(trackReplayList))
        replayList = makeUnique<DisplayList>();

#if !LOG_DISABLED
    size_t i = 0;
#endif
    ReplayResult result;
    for (auto& item : m_items) {
        LOG_WITH_STREAM(DisplayLists, stream << "applying " << i++ << " " << item);

        auto applyResult = applyItem(m_context, m_resourceHeap, m_controlFactory, item, m_options);
        if (applyResult.stopReason) {
            result.reasonForStopping = *applyResult.stopReason;
            result.missingCachedResourceIdentifier = WTFMove(applyResult.resourceIdentifier);
            LOG_WITH_STREAM(DisplayLists, stream << " failed to replay for reason " << result.reasonForStopping << ". Resource " << result.missingCachedResourceIdentifier << " is missing");
            break;
        }

        if (UNLIKELY(trackReplayList))
            replayList->items().append(item);
    }

    result.trackedDisplayList = WTFMove(replayList);
    return result;
}

} // namespace DisplayList
} // namespace WebCore
