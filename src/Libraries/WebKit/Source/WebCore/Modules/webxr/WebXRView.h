/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 17, 2024.
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

#if ENABLE(WEBXR)

#include "JSValueInWrappedObject.h"
#include "WebXRRigidTransform.h"
#include "XREye.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class WebXRFrame;
class WebXRRigidTransform;
class WebXRSession;

class WebXRView : public RefCounted<WebXRView> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(WebXRView, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT static Ref<WebXRView> create(Ref<WebXRFrame>&&, XREye, Ref<WebXRRigidTransform>&&, Ref<Float32Array>&&);
    WEBCORE_EXPORT ~WebXRView();

    const WebXRFrame& frame() const { return m_frame.get(); }
    XREye eye() const { return m_eye; }
    const Float32Array& projectionMatrix() const { return m_projection.get(); }
    const WebXRRigidTransform& transform() const { return m_transform.get(); }

    std::optional<double> recommendedViewportScale() const;
    void requestViewportScale(std::optional<double>);

    double requestedViewportScale() const { return m_requestedViewportScale; }
    bool isViewportModifiable() const { return m_viewportModifiable; }
    void setViewportModifiable(bool modifiable) { m_viewportModifiable = modifiable; }

    JSValueInWrappedObject& cachedProjectionMatrix() { return m_cachedProjectionMatrix; }

private:
    WebXRView(Ref<WebXRFrame>&&, XREye, Ref<WebXRRigidTransform>&&, Ref<Float32Array>&&);

    Ref<WebXRFrame> m_frame;
    XREye m_eye;
    Ref<WebXRRigidTransform> m_transform;
    Ref<Float32Array> m_projection;
    bool m_viewportModifiable { false };
    double m_requestedViewportScale { 1.0 };
    JSValueInWrappedObject m_cachedProjectionMatrix;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
