/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 1, 2023.
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

#if ENABLE(WEBXR_LAYERS)

#include "XRCompositionLayer.h"

namespace WebCore {

class WebXRRigidTransform;
class WebXRSpace;

// https://immersive-web.github.io/layers/#xrequirectlayertype
class XREquirectLayer : public XRCompositionLayer {
public:
    const WebXRSpace& space() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setSpace(WebXRSpace&) { RELEASE_ASSERT_NOT_REACHED(); }
    const WebXRRigidTransform& transform() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setTransform(WebXRRigidTransform&) { RELEASE_ASSERT_NOT_REACHED(); }

    float radius() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setRadius(float) { RELEASE_ASSERT_NOT_REACHED(); }
    float centralHorizontalAngle() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setCentralHorizontalAngle(float) { RELEASE_ASSERT_NOT_REACHED(); }
    float upperVerticalAngle() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setUpperVerticalAngle(float) { RELEASE_ASSERT_NOT_REACHED(); }
    float lowerVerticalAngle() const { RELEASE_ASSERT_NOT_REACHED(); }
    [[noreturn]] void setLowerVerticalAngle(float) { RELEASE_ASSERT_NOT_REACHED(); }
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
