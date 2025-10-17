/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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

#include "ContextDestructionObserver.h"
#include "EventTarget.h"
#include "TransformationMatrix.h"
#include "WebXRSession.h"
#include <wtf/RefCounted.h>

namespace WebCore {

class Document;
class ScriptExecutionContext;
class WebXRRigidTransform;

class WebXRSpace : public EventTarget, public ContextDestructionObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRSpace);
public:
    virtual ~WebXRSpace();

    virtual WebXRSession* session() const = 0;
    virtual std::optional<TransformationMatrix> nativeOrigin() const = 0;
    std::optional<TransformationMatrix> effectiveOrigin() const;
    virtual std::optional<bool> isPositionEmulated() const;

    virtual bool isReferenceSpace() const { return false; }
    virtual bool isBoundedReferenceSpace() const { return false; }
#if ENABLE(WEBXR_HANDS)
    virtual bool isJointSpace() const { return false; }
#endif

protected:
    WebXRSpace(Document&, Ref<WebXRRigidTransform>&&);

    const WebXRRigidTransform& originOffset() const { return m_originOffset.get(); }

    // EventTarget
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

private:
    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::WebXRSpace; }

    Ref<WebXRRigidTransform> m_originOffset;
};

// https://immersive-web.github.io/webxr/#xrsession-viewer-reference-space
// This is a helper class to implement the viewer space owned by a WebXRSession.
// It avoids a circular reference between the session and the reference space.
class WebXRViewerSpace : public RefCounted<WebXRViewerSpace>, public WebXRSpace {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRViewerSpace);
public:
    static Ref< WebXRViewerSpace> create(Document& document, WebXRSession& session)
    {
        return adoptRef(*new WebXRViewerSpace(document, session));
    }
    virtual ~WebXRViewerSpace();

    using RefCounted::ref;
    using RefCounted::deref;

private:
    WebXRViewerSpace(Document&, WebXRSession&);

    WebXRSession* session() const final { return m_session.get(); }
    std::optional<TransformationMatrix> nativeOrigin() const final;

    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    WeakPtr<WebXRSession> m_session;
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_WEBXRSPACE(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
    static bool isType(const WebCore::WebXRSpace& context) { return context.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(WEBXR)
