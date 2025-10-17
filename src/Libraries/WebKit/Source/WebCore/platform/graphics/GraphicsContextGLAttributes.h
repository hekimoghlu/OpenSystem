/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include <optional>

namespace WebCore {

enum class GraphicsContextGLPowerPreference : uint8_t {
    Default,
    LowPower,
    HighPerformance
};

enum class GraphicsContextGLSimulatedCreationFailure : uint8_t {
    None,
    IPCBufferOOM,
    CreationTimeout,
    FailPlatformContextCreation
};

#if PLATFORM(MAC)
using PlatformGPUID = uint64_t;
#endif

struct GraphicsContextGLAttributes {
    bool alpha { true };
    bool depth { true };
    bool stencil { false };
    bool antialias { true };
    bool premultipliedAlpha { true };
    bool preserveDrawingBuffer { false };
    GraphicsContextGLPowerPreference powerPreference { GraphicsContextGLPowerPreference::Default };
    bool isWebGL2 { false };
#if PLATFORM(MAC)
    PlatformGPUID windowGPUID { 0 };
#endif
#if ENABLE(WEBXR)
    bool xrCompatible { false };
#endif
    using SimulatedCreationFailure = GraphicsContextGLSimulatedCreationFailure;
    SimulatedCreationFailure failContextCreationForTesting { SimulatedCreationFailure::None };
};
}

#endif // ENABLE(WEBGL)
