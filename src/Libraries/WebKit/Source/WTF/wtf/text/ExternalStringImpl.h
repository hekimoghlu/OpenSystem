/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

#include <wtf/Function.h>
#include <wtf/text/StringImpl.h>

namespace WTF {

class ExternalStringImpl;

using ExternalStringImplFreeFunction = Function<void(ExternalStringImpl*, void*, unsigned)>;

class SUPPRESS_REFCOUNTED_WITHOUT_VIRTUAL_DESTRUCTOR ExternalStringImpl final : public StringImpl {
public:
    WTF_EXPORT_PRIVATE static Ref<ExternalStringImpl> create(std::span<const LChar> characters, ExternalStringImplFreeFunction&&);
    WTF_EXPORT_PRIVATE static Ref<ExternalStringImpl> create(std::span<const UChar> characters, ExternalStringImplFreeFunction&&);

private:
    friend class StringImpl;

    ExternalStringImpl(std::span<const LChar> characters, ExternalStringImplFreeFunction&&);
    ExternalStringImpl(std::span<const UChar> characters, ExternalStringImplFreeFunction&&);

    inline void freeExternalBuffer(void* buffer, unsigned bufferSize);

    ExternalStringImplFreeFunction m_free;
};

ALWAYS_INLINE void ExternalStringImpl::freeExternalBuffer(void* buffer, unsigned bufferSize)
{
    m_free(this, buffer, bufferSize);
}

} // namespace WTF

using WTF::ExternalStringImpl;
