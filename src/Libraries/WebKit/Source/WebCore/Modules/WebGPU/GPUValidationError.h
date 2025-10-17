/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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

#include "WebGPUValidationError.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GPUValidationError : public RefCounted<GPUValidationError> {
public:
    static Ref<GPUValidationError> create(String&& message)
    {
        return adoptRef(*new GPUValidationError(WTFMove(message)));
    }

    static Ref<GPUValidationError> create(Ref<WebGPU::ValidationError>&& backing)
    {
        return adoptRef(*new GPUValidationError(WTFMove(backing)));
    }

    const String& message() const;

    WebGPU::ValidationError* backing() { return m_backing.get(); }
    const WebGPU::ValidationError* backing() const { return m_backing.get(); }
    String stack() const { return "_"_s; }

private:
    GPUValidationError(String&& message)
        : m_message(WTFMove(message))
    {
    }

    GPUValidationError(Ref<WebGPU::ValidationError>&& backing)
        : m_backing(WTFMove(backing))
    {
    }

    String m_message;
    RefPtr<WebGPU::ValidationError> m_backing;
};

}
