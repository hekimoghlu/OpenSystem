/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
#include "GPUBasedCanvasRenderingContext.h"

#include "HTMLCanvasElement.h"
#include "RenderBox.h"

namespace WebCore {

GPUBasedCanvasRenderingContext::GPUBasedCanvasRenderingContext(CanvasBase& canvas, CanvasRenderingContext::Type type)
    : CanvasRenderingContext(canvas, type)
    , ActiveDOMObject(canvas.scriptExecutionContext())
{
    ASSERT(isGPUBased());
}

HTMLCanvasElement* GPUBasedCanvasRenderingContext::htmlCanvas() const
{
    return dynamicDowncast<HTMLCanvasElement>(canvasBase());
}

void GPUBasedCanvasRenderingContext::markCanvasChanged()
{
    auto& canvas = canvasBase();
    canvas.didDraw(FloatRect { { }, canvas.size() }, ShouldApplyPostProcessingToDirtyRect::No);
}

} // namespace WebCore
