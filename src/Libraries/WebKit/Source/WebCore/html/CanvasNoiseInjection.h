/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#include "IntRect.h"

namespace WebCore {

class CanvasBase;
class ImageBuffer;
class PixelBuffer;

enum class CanvasNoiseInjectionPostProcessArea : bool { DirtyRect, FullBuffer };
using NoiseInjectionHashSalt = uint64_t;

class CanvasNoiseInjection {
public:
    void postProcessDirtyCanvasBuffer(ImageBuffer*, NoiseInjectionHashSalt, CanvasNoiseInjectionPostProcessArea = CanvasNoiseInjectionPostProcessArea::DirtyRect);
    bool postProcessPixelBufferResults(PixelBuffer&, NoiseInjectionHashSalt) const;
    void updateDirtyRect(const IntRect&);
    void clearDirtyRect();
    bool haveDirtyRects() const { return !m_postProcessDirtyRect.isEmpty(); }

private:
    IntRect m_postProcessDirtyRect;
};
} // namespace WebCore
