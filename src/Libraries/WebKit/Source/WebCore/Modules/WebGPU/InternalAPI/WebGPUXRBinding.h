/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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

#include "WebGPUXREye.h"
#include "WebGPUXRProjectionLayer.h"
#include "WebGPUXRSubImage.h"

#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class WebXRFrame;
}

namespace WebCore::WebGPU {

class Device;
class XRGPUSubImage;
class XRProjectionLayer;
class XRView;

class XRBinding : public RefCountedAndCanMakeWeakPtr<XRBinding> {
public:
    virtual ~XRBinding() = default;

    virtual RefPtr<XRProjectionLayer> createProjectionLayer(const XRProjectionLayerInit&) = 0;
    virtual RefPtr<XRSubImage> getSubImage(XRProjectionLayer&, WebCore::WebXRFrame&, std::optional<XREye>/* = "none"*/) = 0;
    virtual RefPtr<XRSubImage> getViewSubImage(XRProjectionLayer&) = 0;
    virtual TextureFormat getPreferredColorFormat() = 0;

protected:
    XRBinding() = default;

private:
    XRBinding(const XRBinding&) = delete;
    XRBinding(XRBinding&&) = delete;
    XRBinding& operator=(const XRBinding&) = delete;
    XRBinding& operator=(XRBinding&&) = delete;
};

} // namespace WebCore::WebGPU
