/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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

#include "APIObject.h"
#include <wtf/Function.h>
#include <wtf/StdLibExtras.h>
#include <wtf/Vector.h>

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
#endif

OBJC_CLASS NSData;

namespace API {

class Data : public ObjectImpl<API::Object::Type::Data> {
public:
    using FreeDataFunction = WTF::Function<void()>;

    static Ref<Data> createWithoutCopying(std::span<const uint8_t> bytes, FreeDataFunction&& freeDataFunction)
    {
        return adoptRef(*new Data(bytes, WTFMove(freeDataFunction)));
    }

    static Ref<Data> create(std::span<const uint8_t> bytes)
    {
        MallocSpan<uint8_t> copiedBytes;

        if (!bytes.empty()) {
            copiedBytes = MallocSpan<uint8_t>::malloc(bytes.size_bytes());
            memcpySpan(copiedBytes.mutableSpan(), bytes);
        }

        auto data = copiedBytes.span();
        return createWithoutCopying(data, [copiedBytes = WTFMove(copiedBytes)] () { });
    }
    
    static Ref<Data> create(const Vector<unsigned char>& buffer)
    {
        return create(buffer.span());
    }

    static Ref<Data> create(Vector<unsigned char>&& vector)
    {
        auto buffer = vector.releaseBuffer();
        auto span = buffer.span();
        return createWithoutCopying(span, [buffer = WTFMove(buffer)] { });
    }

#if PLATFORM(COCOA)
    static Ref<Data> createWithoutCopying(RetainPtr<NSData>);
#endif

    ~Data()
    {
        m_freeDataFunction();
    }

    size_t size() const { return m_span.size(); }
    std::span<const uint8_t> span() const { return m_span; }

private:
    Data(std::span<const uint8_t> span, FreeDataFunction&& freeDataFunction)
        : m_span(span)
        , m_freeDataFunction(WTFMove(freeDataFunction))
    {
    }

    std::span<const uint8_t> m_span;
    FreeDataFunction m_freeDataFunction;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(Data);
