/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore::WebGPU {

class Device;
class Texture;

class XRSubImage : public RefCountedAndCanMakeWeakPtr<XRSubImage> {
public:
    virtual ~XRSubImage() = default;

    virtual RefPtr<Texture> colorTexture() = 0;
    virtual RefPtr<Texture> depthStencilTexture() = 0;
    virtual RefPtr<Texture> motionVectorTexture() = 0;

protected:
    XRSubImage() = default;

private:
    XRSubImage(const XRSubImage&) = delete;
    XRSubImage(XRSubImage&&) = delete;
    XRSubImage& operator=(const XRSubImage&) = delete;
    XRSubImage& operator=(XRSubImage&&) = delete;
};

} // namespace WebCore::WebGPU
