/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

using CVPixelBufferRef = struct __CVBuffer*;

namespace WebCore::WebGPU {

class ExternalTexture : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<ExternalTexture> {
public:
    virtual ~ExternalTexture() = default;

    String label() const { return m_label; }

    void setLabel(String&& label)
    {
        m_label = WTFMove(label);
        setLabelInternal(m_label);
    }
    virtual void destroy() = 0;
    virtual void undestroy() = 0;
    virtual void updateExternalTexture(CVPixelBufferRef) = 0;

protected:
    ExternalTexture() = default;

private:
    ExternalTexture(const ExternalTexture&) = delete;
    ExternalTexture(ExternalTexture&&) = delete;
    ExternalTexture& operator=(const ExternalTexture&) = delete;
    ExternalTexture& operator=(ExternalTexture&&) = delete;

    virtual void setLabelInternal(const String&) = 0;

    String m_label;
};

} // namespace WebCore::WebGPU
