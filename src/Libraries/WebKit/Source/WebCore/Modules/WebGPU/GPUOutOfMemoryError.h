/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include "WebGPUOutOfMemoryError.h"
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUOutOfMemoryError : public RefCounted<GPUOutOfMemoryError> {
public:
    static Ref<GPUOutOfMemoryError> create(String&& message)
    {
        return adoptRef(*new GPUOutOfMemoryError(WTFMove(message)));
    }

    static Ref<GPUOutOfMemoryError> create(Ref<WebGPU::OutOfMemoryError>&& backing)
    {
        return adoptRef(*new GPUOutOfMemoryError(WTFMove(backing)));
    }

    const String& message() const { return m_message; }

    WebGPU::OutOfMemoryError* backing() { return m_backing.get(); }
    const WebGPU::OutOfMemoryError* backing() const { return m_backing.get(); }

private:
    GPUOutOfMemoryError(String&& message)
        : m_message(WTFMove(message))
    {
    }

    GPUOutOfMemoryError(Ref<WebGPU::OutOfMemoryError>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    String m_message;
    RefPtr<WebGPU::OutOfMemoryError> m_backing;
};

}
