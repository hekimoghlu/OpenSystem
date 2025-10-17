/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#include "PerformanceLoggingClient.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PerformanceLoggingClient);

String PerformanceLoggingClient::synchronousScrollingReasonsAsString(OptionSet<SynchronousScrollingReason> reasons)
{
    if (reasons.isEmpty())
        return emptyString();

    auto string = makeString(reasons.contains(SynchronousScrollingReason::ForcedOnMainThread) ? "forced,"_s : ""_s,
        reasons.contains(SynchronousScrollingReason::HasSlowRepaintObjects) ? "slow-repaint objects,"_s : ""_s,
        reasons.contains(SynchronousScrollingReason::HasViewportConstrainedObjectsWithoutSupportingFixedLayers) ? "viewport-constrained objects,"_s : ""_s,
        reasons.contains(SynchronousScrollingReason::HasNonLayerViewportConstrainedObjects) ? "non-layer viewport-constrained objects,"_s : ""_s,
        reasons.contains(SynchronousScrollingReason::IsImageDocument) ? "image document,"_s : ""_s);

    // Strip the trailing comma.
    return string.left(string.length() - 1);
}

} // namespace WebCore
