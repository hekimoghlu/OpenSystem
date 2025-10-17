/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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

#include "ArrayBufferView.h"
#include <wtf/FlipBytes.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class DataView final : public ArrayBufferView {
public:
    JS_EXPORT_PRIVATE static Ref<DataView> create(RefPtr<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> length);
    static Ref<DataView> create(RefPtr<ArrayBuffer>&&);

    JSArrayBufferView* wrapImpl(JSGlobalObject* lexicalGlobalObject, JSGlobalObject* globalObject);
    
    template<typename T>
    T get(size_t offset, bool littleEndian, bool* status = nullptr)
    {
        if (status) {
            if (offset + sizeof(T) > byteLength()) {
                *status = false;
                return T();
            }
            *status = true;
        } else
            RELEASE_ASSERT(offset + sizeof(T) <= byteLength());
        return flipBytesIfLittleEndian(
            *reinterpret_cast<T*>(static_cast<uint8_t*>(baseAddress()) + offset),
            littleEndian);
    }
    
    template<typename T>
    T read(size_t& offset, bool littleEndian, bool* status = nullptr)
    {
        T result = this->template get<T>(offset, littleEndian, status);
        if (!status || *status)
            offset += sizeof(T);
        return result;
    }
    
    template<typename T>
    void set(size_t offset, T value, bool littleEndian, bool* status = nullptr)
    {
        if (status) {
            if (offset + sizeof(T) > byteLength()) {
                *status = false;
                return;
            }
            *status = true;
        } else
            RELEASE_ASSERT(offset + sizeof(T) <= byteLength());
        *reinterpret_cast<T*>(static_cast<uint8_t*>(baseAddress()) + offset) =
            flipBytesIfLittleEndian(value, littleEndian);
    }

    JS_EXPORT_PRIVATE static RefPtr<DataView> wrappedAs(Ref<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> length);

private:
    DataView(RefPtr<ArrayBuffer>&&, size_t byteOffset, std::optional<size_t> byteLength);
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
