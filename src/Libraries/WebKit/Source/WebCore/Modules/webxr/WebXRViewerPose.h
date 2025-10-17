/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 20, 2023.
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

#include "WebXRPose.h"
#include "WebXRView.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class WebXRViewerPose : public WebXRPose {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRViewerPose);
public:
    static Ref<WebXRViewerPose> create(Ref<WebXRRigidTransform>&&, bool emulatedPosition);
    virtual ~WebXRViewerPose();

    const Vector<Ref<WebXRView>>& views() const;
    void setViews(Vector<Ref<WebXRView>>&&);

    JSValueInWrappedObject& cachedViews() { return m_cachedViews; }

private:
    WebXRViewerPose(Ref<WebXRRigidTransform>&&, bool emulatedPosition);

    bool isViewerPose() const final { return true; }

    Vector<Ref<WebXRView>> m_views;
    JSValueInWrappedObject m_cachedViews;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_WEBXRPOSE(WebXRViewerPose, isViewerPose())

#endif // ENABLE(WEBXR)
