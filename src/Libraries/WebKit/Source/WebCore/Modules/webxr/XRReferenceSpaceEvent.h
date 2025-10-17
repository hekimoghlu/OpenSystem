/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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

#include "Event.h"
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class WebXRReferenceSpace;
class WebXRRigidTransform;

class XRReferenceSpaceEvent : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRReferenceSpaceEvent);
public:
    struct Init : EventInit {
        RefPtr<WebXRReferenceSpace> referenceSpace;
        RefPtr<WebXRRigidTransform> transform;
    };

    static Ref<XRReferenceSpaceEvent> create(const AtomString&, const Init&, IsTrusted = IsTrusted::No);
    virtual ~XRReferenceSpaceEvent();

    const WebXRReferenceSpace& referenceSpace() const;
    WebXRRigidTransform* transform() const;

private:
    XRReferenceSpaceEvent(const AtomString&, const Init&, IsTrusted);

    RefPtr<WebXRReferenceSpace> m_referenceSpace;
    RefPtr<WebXRRigidTransform> m_transform;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
