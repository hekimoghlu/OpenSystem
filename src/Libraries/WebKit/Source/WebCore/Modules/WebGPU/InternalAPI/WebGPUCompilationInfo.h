/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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

#include "WebGPUCompilationMessage.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore::WebGPU {

class CompilationInfo final : public RefCounted<CompilationInfo> {
public:
    static Ref<CompilationInfo> create(Vector<Ref<CompilationMessage>>&& messages)
    {
        return adoptRef(*new CompilationInfo(WTFMove(messages)));
    }

    const Vector<Ref<CompilationMessage>>& messages() const { return m_messages; }

protected:
    CompilationInfo(Vector<Ref<CompilationMessage>>&& messages)
        : m_messages(WTFMove(messages))
    {
    }

private:
    CompilationInfo(const CompilationInfo&) = delete;
    CompilationInfo(CompilationInfo&&) = delete;
    CompilationInfo& operator=(const CompilationInfo&) = delete;
    CompilationInfo& operator=(CompilationInfo&&) = delete;

    Vector<Ref<CompilationMessage>> m_messages;
};

} // namespace WebCore::WebGPU
