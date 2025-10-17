/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 23, 2023.
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
#include "PerformanceNavigation.h"

#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "FrameLoaderTypes.h"
#include "LocalFrame.h"

namespace WebCore {

PerformanceNavigation::PerformanceNavigation(LocalDOMWindow* window)
    : LocalDOMWindowProperty(window)
{
}

unsigned short PerformanceNavigation::type() const
{
    auto* frame = this->frame();
    if (!frame)
        return TYPE_NAVIGATE;

    DocumentLoader* documentLoader = frame->loader().documentLoader();
    if (!documentLoader)
        return TYPE_NAVIGATE;

    WebCore::NavigationType navigationType = documentLoader->triggeringAction().type();
    switch (navigationType) {
    case NavigationType::Reload:
        return TYPE_RELOAD;
    case NavigationType::BackForward:
        return TYPE_BACK_FORWARD;
    default:
        return TYPE_NAVIGATE;
    }
}

unsigned short PerformanceNavigation::redirectCount() const
{
    RefPtr frame = this->frame();
    if (!frame)
        return 0;

    RefPtr loader = frame->loader().documentLoader();
    if (!loader)
        return 0;

    auto* metrics = loader->response().deprecatedNetworkLoadMetricsOrNull();
    if (!metrics)
        return 0;

    if (metrics->failsTAOCheck)
        return 0;

    return metrics->redirectCount;
}

} // namespace WebCore
