/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUPtr.h"
#include "WebGPURenderBundle.h"
#include <WebGPU/WebGPU.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;

class RenderBundleImpl final : public RenderBundle {
    WTF_MAKE_TZONE_ALLOCATED(RenderBundleImpl);
public:
    static Ref<RenderBundleImpl> create(WebGPUPtr<WGPURenderBundle>&& renderBundle, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new RenderBundleImpl(WTFMove(renderBundle), convertToBackingContext));
    }

    virtual ~RenderBundleImpl();

private:
    friend class DowncastConvertToBackingContext;

    RenderBundleImpl(WebGPUPtr<WGPURenderBundle>&&, ConvertToBackingContext&);

    RenderBundleImpl(const RenderBundleImpl&) = delete;
    RenderBundleImpl(RenderBundleImpl&&) = delete;
    RenderBundleImpl& operator=(const RenderBundleImpl&) = delete;
    RenderBundleImpl& operator=(RenderBundleImpl&&) = delete;

    WGPURenderBundle backing() const { return m_backing.get(); }

    void setLabelInternal(const String&) final;

    WebGPUPtr<WGPURenderBundle> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
