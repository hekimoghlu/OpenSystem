/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#include "WebXRReferenceSpace.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class DOMPointReadOnly;

class WebXRBoundedReferenceSpace final : public WebXRReferenceSpace {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRBoundedReferenceSpace);
public:
    static Ref<WebXRBoundedReferenceSpace> create(Document&, WebXRSession&, XRReferenceSpaceType);
    static Ref<WebXRBoundedReferenceSpace> create(Document&, WebXRSession&, Ref<WebXRRigidTransform>&&, XRReferenceSpaceType);

    virtual ~WebXRBoundedReferenceSpace();

    std::optional<TransformationMatrix> nativeOrigin() const final;
    const Vector<Ref<DOMPointReadOnly>>& boundsGeometry();
    ExceptionOr<Ref<WebXRReferenceSpace>> getOffsetReferenceSpace(const WebXRRigidTransform&) final;

private:
    WebXRBoundedReferenceSpace(Document&, WebXRSession&, Ref<WebXRRigidTransform>&&, XRReferenceSpaceType);

    bool isBoundedReferenceSpace() const final { return true; }

    void updateIfNeeded();
    float quantize(float);

    Vector<Ref<DOMPointReadOnly>> m_boundsGeometry;
    int m_lastUpdateId { -1 };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WebXRBoundedReferenceSpace)
    static bool isType(const WebCore::WebXRReferenceSpace& element) { return element.isBoundedReferenceSpace(); }
    static bool isType(const WebCore::WebXRSpace& element)
    {
        auto* referenceSpace = dynamicDowncast<WebCore::WebXRReferenceSpace>(element);
        return referenceSpace && isType(*referenceSpace);
    }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEBXR)
