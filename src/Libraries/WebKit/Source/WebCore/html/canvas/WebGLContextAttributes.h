/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#if ENABLE(WEBGL)

#include "GraphicsContextGLAttributes.h"

namespace WebCore {

enum class WebGLVersion : uint8_t {
    WebGL1,
    WebGL2
};

using WebGLPowerPreference = GraphicsContextGLPowerPreference;

using WebGLContextSimulatedCreationFailure = GraphicsContextGLSimulatedCreationFailure;

struct WebGLContextAttributes {
    bool alpha { true };
    bool depth { true };
    bool stencil { false };
    bool antialias { true };
    bool premultipliedAlpha { true };
    bool preserveDrawingBuffer { false };
    using PowerPreference = WebGLPowerPreference;
    PowerPreference powerPreference { PowerPreference::Default };
    bool failIfMajorPerformanceCaveat { false };
#if ENABLE(WEBXR)
    bool xrCompatible { false };
#endif
    using SimulatedCreationFailure = WebGLContextSimulatedCreationFailure;
    SimulatedCreationFailure failContextCreationForTesting { SimulatedCreationFailure::None };
};

} // namespace WebCore

#endif
