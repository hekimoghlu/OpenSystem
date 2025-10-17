/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

#include "XREye.h"

#include <ExceptionOr.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class WebGL2RenderingContext;
class WebGLRenderingContext;
class WebXRFrame;
class WebXRSession;
class WebXRView;
class XRCompositionLayer;
class XRCubeLayer;
class XRCylinderLayer;
class XREquirectLayer;
class XRProjectionLayer;
class XRQuadLayer;
class XRWebGLSubImage;

struct XRCubeLayerInit;
struct XRCylinderLayerInit;
struct XREquirectLayerInit;
struct XRProjectionLayerInit;
struct XRQuadLayerInit;

// https://immersive-web.github.io/layers/#XRWebGLBindingtype
class XRWebGLBinding : public RefCounted<XRWebGLBinding> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRWebGLBinding);
public:

    using WebXRWebGLRenderingContext = std::variant<
        RefPtr<WebGLRenderingContext>,
        RefPtr<WebGL2RenderingContext>
    >;

    static ExceptionOr<Ref<XRWebGLBinding>> create(Ref<WebXRSession>&&, WebXRWebGLRenderingContext&&);
    ~XRWebGLBinding() = default;

    double nativeProjectionScaleFactor() const { RELEASE_ASSERT_NOT_REACHED(); }
    bool usesDepthValues() const { RELEASE_ASSERT_NOT_REACHED(); }

    ExceptionOr<Ref<XRProjectionLayer>> createProjectionLayer(const XRProjectionLayerInit&) { RELEASE_ASSERT_NOT_REACHED(); }
    ExceptionOr<Ref<XRQuadLayer>> createQuadLayer(const XRQuadLayerInit&) { RELEASE_ASSERT_NOT_REACHED(); }
    ExceptionOr<Ref<XRCylinderLayer>> createCylinderLayer(const XRCylinderLayerInit&) { RELEASE_ASSERT_NOT_REACHED(); }
    ExceptionOr<Ref<XREquirectLayer>> createEquirectLayer(const XREquirectLayerInit&) { RELEASE_ASSERT_NOT_REACHED(); }
    ExceptionOr<Ref<XRCubeLayer>> createCubeLayer(const XRCubeLayerInit&) { RELEASE_ASSERT_NOT_REACHED(); }

    ExceptionOr<Ref<XRWebGLSubImage>> getSubImage(const XRCompositionLayer&, const WebXRFrame&, XREye) { RELEASE_ASSERT_NOT_REACHED(); }
    ExceptionOr<Ref<XRWebGLSubImage>> getViewSubImage(const XRProjectionLayer&, const WebXRView&) { RELEASE_ASSERT_NOT_REACHED(); }
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
