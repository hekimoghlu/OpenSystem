/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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

#include "WebXRSpace.h"
#include "XRReferenceSpaceType.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class WebXRRigidTransform;
class WebXRSession;

class WebXRReferenceSpace : public RefCounted<WebXRReferenceSpace>, public WebXRSpace {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRReferenceSpace);
public:
    static Ref<WebXRReferenceSpace> create(Document&, WebXRSession&, XRReferenceSpaceType);
    static Ref<WebXRReferenceSpace> create(Document&, WebXRSession&, Ref<WebXRRigidTransform>&&, XRReferenceSpaceType);

    virtual ~WebXRReferenceSpace();

    using RefCounted<WebXRReferenceSpace>::ref;
    using RefCounted<WebXRReferenceSpace>::deref;

    WebXRSession* session() const final { return m_session.get(); }
    std::optional<TransformationMatrix> nativeOrigin() const override;
    virtual ExceptionOr<Ref<WebXRReferenceSpace>> getOffsetReferenceSpace(const WebXRRigidTransform&);
    XRReferenceSpaceType type() const { return m_type; }

protected:
    WebXRReferenceSpace(Document&, WebXRSession&, Ref<WebXRRigidTransform>&&, XRReferenceSpaceType);

    bool isReferenceSpace() const final { return true; }

    std::optional<TransformationMatrix> floorOriginTransform() const;

    WeakPtr<WebXRSession> m_session;
    XRReferenceSpaceType m_type;

private:
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_WEBXRSPACE(WebXRReferenceSpace, isReferenceSpace())

#endif // ENABLE(WEBXR)
