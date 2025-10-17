/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#include "Event.h"
#include "WebXRLayer.h"

namespace WebCore {

class XRLayerEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XRLayerEvent);
public:
    struct Init : EventInit {
        Init() = default;
        Init(RefPtr<WebXRLayer>&& layer)
            : EventInit()
            , layer(WTFMove(layer))
        { }
        RefPtr<WebXRLayer> layer;
    };

    static Ref<XRLayerEvent> create(const AtomString&, const Init&, IsTrusted = IsTrusted::No);
    virtual ~XRLayerEvent();

    const WebXRLayer& layer() const;

protected:
    XRLayerEvent(const AtomString&, const Init&, IsTrusted);

    // Event.
    EventInterfaceType eventInterfaceType() const;

private:
    RefPtr<WebXRLayer> m_layer;
};

} // namespace WebCore

#endif // ENABLE(WEBXR_LAYERS)
