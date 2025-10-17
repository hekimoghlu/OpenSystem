/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 23, 2022.
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

#include "DisplayList.h"
#include "DisplayListItem.h"
#include <wtf/Noncopyable.h>

namespace WebCore {

class ControlFactory;
class FloatRect;
class GraphicsContext;

namespace DisplayList {

class DisplayList;
class ResourceHeap;

struct ReplayResult {
    std::unique_ptr<DisplayList> trackedDisplayList;
    std::optional<RenderingResourceIdentifier> missingCachedResourceIdentifier;
    StopReplayReason reasonForStopping { StopReplayReason::ReplayedAllItems };
};

class Replayer {
    WTF_MAKE_NONCOPYABLE(Replayer);
public:
    WEBCORE_EXPORT Replayer(GraphicsContext&, const DisplayList&);
    WEBCORE_EXPORT Replayer(GraphicsContext&, const Vector<Item>&, const ResourceHeap&, ControlFactory&, OptionSet<ReplayOption> = { });
    ~Replayer() = default;

    WEBCORE_EXPORT ReplayResult replay(const FloatRect& initialClip = { }, bool trackReplayList = false);

private:
    GraphicsContext& m_context;
    const Vector<Item>& m_items;
    const ResourceHeap& m_resourceHeap;
    Ref<ControlFactory> m_controlFactory;
    OptionSet<ReplayOption> m_options;
};

} // namespace DisplayList
} // namespace WebCore
