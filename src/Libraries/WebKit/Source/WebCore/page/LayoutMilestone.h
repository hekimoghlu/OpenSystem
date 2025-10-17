/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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

#include <wtf/OptionSet.h>

namespace WebCore {

// FIXME: Some of these milestones are about layout, and others are about painting.
// We should either re-name them to something more generic, or split them into
// two enums -- one for painting and one for layout.
enum class LayoutMilestone : uint16_t {
    DidFirstLayout                                      = 1 << 0,
    DidFirstVisuallyNonEmptyLayout                      = 1 << 1,
    DidHitRelevantRepaintedObjectsAreaThreshold         = 1 << 2,
    DidFirstLayoutAfterSuppressedIncrementalRendering   = 1 << 4,
    DidFirstPaintAfterSuppressedIncrementalRendering    = 1 << 5,
    ReachedSessionRestorationRenderTreeSizeThreshold    = 1 << 6, // FIXME: only implemented by WK2 currently.
    DidRenderSignificantAmountOfText                    = 1 << 7,
    DidFirstMeaningfulPaint                             = 1 << 8,
};

} // namespace WebCore
