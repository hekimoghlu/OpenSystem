/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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

#include "WKPageRenderingProgressEvents.h"

#include <WebCore/LayoutMilestone.h>

static inline WKPageRenderingProgressEvents pageRenderingProgressEvents(OptionSet<WebCore::LayoutMilestone> milestones)
{
    WKPageRenderingProgressEvents events = 0;
    
    if (milestones & WebCore::LayoutMilestone::DidFirstLayout)
        events |= WKPageRenderingProgressEventFirstLayout;
    
    if (milestones & WebCore::LayoutMilestone::DidFirstVisuallyNonEmptyLayout)
        events |= WKPageRenderingProgressEventFirstVisuallyNonEmptyLayout;
    
    if (milestones & WebCore::LayoutMilestone::DidHitRelevantRepaintedObjectsAreaThreshold)
        events |= WKPageRenderingProgressEventFirstPaintWithSignificantArea;
    
    if (milestones & WebCore::LayoutMilestone::ReachedSessionRestorationRenderTreeSizeThreshold)
        events |= WKPageRenderingProgressEventReachedSessionRestorationRenderTreeSizeThreshold;
    
    if (milestones & WebCore::LayoutMilestone::DidFirstLayoutAfterSuppressedIncrementalRendering)
        events |= WKPageRenderingProgressEventFirstLayoutAfterSuppressedIncrementalRendering;

    if (milestones & WebCore::LayoutMilestone::DidFirstPaintAfterSuppressedIncrementalRendering)
        events |= WKPageRenderingProgressEventFirstPaintAfterSuppressedIncrementalRendering;

    if (milestones & WebCore::LayoutMilestone::DidFirstMeaningfulPaint)
        events |= WKPageRenderingProgressEventFirstMeaningfulPaint;

    return events;
}
