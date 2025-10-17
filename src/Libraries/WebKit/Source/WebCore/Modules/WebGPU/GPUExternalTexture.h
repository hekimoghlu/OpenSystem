/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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

#include "WebGPUExternalTexture.h"
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUExternalTexture : public RefCountedAndCanMakeWeakPtr<GPUExternalTexture> {
public:
    static Ref<GPUExternalTexture> create(Ref<WebGPU::ExternalTexture>&& backing)
    {
        return adoptRef(*new GPUExternalTexture(WTFMove(backing)));
    }

    String label() const;
    void setLabel(String&&);

    WebGPU::ExternalTexture& backing() { return m_backing; }
    const WebGPU::ExternalTexture& backing() const { return m_backing; }
    void destroy();
    void undestroy();

private:
    GPUExternalTexture(Ref<WebGPU::ExternalTexture>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    Ref<WebGPU::ExternalTexture> m_backing;
};

}
