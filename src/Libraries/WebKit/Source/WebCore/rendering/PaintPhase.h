/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

namespace WebCore {

/*
 *  The painting of a layer occurs in three distinct phases.  Each phase involves
 *  a recursive descent into the layer's render objects. The first phase is the background phase.
 *  The backgrounds and borders of all blocks are painted.  Inlines are not painted at all.
 *  Floats must paint above block backgrounds but entirely below inline content that can overlap them.
 *  In the foreground phase, all inlines are fully painted.  Inline replaced elements will get all
 *  three phases invoked on them during this phase.
 */

enum class PaintPhase : uint16_t {
    BlockBackground          = 0,
    ChildBlockBackground     = 1 << 0,
    ChildBlockBackgrounds    = 1 << 1,
    Float                    = 1 << 2,
    Foreground               = 1 << 3,
    Outline                  = 1 << 4,
    ChildOutlines            = 1 << 5,
    SelfOutline              = 1 << 6,
    Selection                = 1 << 7,
    CollapsedTableBorders    = 1 << 8,
    TextClip                 = 1 << 9,
    Mask                     = 1 << 10,
    ClippingMask             = 1 << 11,
    EventRegion              = 1 << 12,
    Accessibility            = 1 << 13,
};

enum class PaintBehavior : uint32_t {
    Normal                              = 0,
    SelectionOnly                       = 1 << 0,
    SkipSelectionHighlight              = 1 << 1,
    ForceBlackText                      = 1 << 2,
    ForceWhiteText                      = 1 << 3,
    ForceBlackBorder                    = 1 << 4,
    RenderingSVGClipOrMask              = 1 << 5,
    SkipRootBackground                  = 1 << 6,
    RootBackgroundOnly                  = 1 << 7,
    SelectionAndBackgroundsOnly         = 1 << 8,
    ExcludeSelection                    = 1 << 9,
    FlattenCompositingLayers            = 1 << 10, // Paint doesn't stop at compositing layer boundaries.
    ForceSynchronousImageDecode         = 1 << 11, // Paint should always complete image decoding of painted images.
    DefaultAsynchronousImageDecode      = 1 << 12, // Paint should always start asynchronous image decode of painted images, unless otherwise specified.
    CompositedOverflowScrollContent     = 1 << 13,
    AnnotateLinks                       = 1 << 14, // Collect all renderers with links to annotate their URLs (e.g. PDFs)
    EventRegionIncludeForeground        = 1 << 15, // FIXME: Event region painting should use paint phases.
    EventRegionIncludeBackground        = 1 << 16,
    Snapshotting                        = 1 << 17, // Paint is updating external backing store and visits all content, including composited content and always completes image decoding of painted images. FIXME: Will be removed.
    DontShowVisitedLinks                = 1 << 18,
    ExcludeReplacedContent              = 1 << 19,
};

} // namespace WebCore
