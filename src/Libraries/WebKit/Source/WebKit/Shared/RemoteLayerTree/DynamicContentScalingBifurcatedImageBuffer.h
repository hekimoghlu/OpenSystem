/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)

#include <WebCore/ImageBuffer.h>
#include <WebCore/ImageBufferCGBackend.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class BifurcatedGraphicsContext;
class DynamicContentScalingDisplayList;
}

namespace WebKit {

class DynamicContentScalingImageBufferBackend;

// Ideally this would be a generic "BifurcatedImageBuffer", but it is
// currently insufficiently general (e.g. needs to support bifurcating
// the context flush, etc.).

class DynamicContentScalingBifurcatedImageBuffer : public WebCore::ImageBuffer {
    WTF_MAKE_TZONE_ALLOCATED(DynamicContentScalingBifurcatedImageBuffer);
public:
    DynamicContentScalingBifurcatedImageBuffer(WebCore::ImageBufferParameters, const WebCore::ImageBufferBackend::Info&, const WebCore::ImageBufferCreationContext&, std::unique_ptr<WebCore::ImageBufferBackend>&& = nullptr, WebCore::RenderingResourceIdentifier = WebCore::RenderingResourceIdentifier::generate());

    WebCore::GraphicsContext& context() const final;

protected:
    std::optional<WebCore::DynamicContentScalingDisplayList> dynamicContentScalingDisplayList() final;

    void releaseGraphicsContext() final;

    mutable std::unique_ptr<WebCore::BifurcatedGraphicsContext> m_context;
    std::unique_ptr<DynamicContentScalingImageBufferBackend> m_dynamicContentScalingBackend;
};

}

#endif // ENABLE(RE_DYNAMIC_CONTENT_SCALING)
