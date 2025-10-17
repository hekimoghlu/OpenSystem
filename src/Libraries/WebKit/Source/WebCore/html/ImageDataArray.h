/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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

#include "ImageDataStorageFormat.h"
#include <JavaScriptCore/Float16Array.h>
#include <JavaScriptCore/Uint8ClampedArray.h>
#include <optional>
#include <variant>
#include <wtf/JSONValues.h>

namespace WebCore {

class ImageDataArray {
public:
    static constexpr bool isSupported(JSC::TypedArrayType type) { return !!toImageDataStorageFormat(type); }
    static bool isSupported(const JSC::ArrayBufferView&);

    ImageDataArray(Ref<JSC::Uint8ClampedArray>&&);
    ImageDataArray(Ref<JSC::Float16Array>&&);

    static std::optional<ImageDataArray> tryCreate(size_t, ImageDataStorageFormat, std::span<const uint8_t> = { });

    ImageDataStorageFormat storageFormat() const;
    size_t length() const;

    JSC::ArrayBufferView& arrayBufferView() const { return m_arrayBufferView.get(); }
    Ref<JSC::ArrayBufferView> protectedArrayBufferView() const { return m_arrayBufferView; }
    auto byteLength() const { return protectedArrayBufferView()->byteLength(); }
    auto isDetached() const { return protectedArrayBufferView()->isDetached(); }
    auto span() const { return protectedArrayBufferView()->span(); }

    Ref<JSC::Uint8ClampedArray> asUint8ClampedArray() const;
    Ref<JSC::Float16Array> asFloat16Array() const;

    Ref<JSON::Value> copyToJSONArray() const;

private:
    ImageDataArray(Ref<JSC::ArrayBufferView>&&);

    // Needed by `toJS<IDLUnion<IDLUint8ClampedArray, ...>, const ImageDataArray&>()`
    template<typename IDL, bool needsState, bool needsGlobalObject> friend struct JSConverterOverloader;
    using Variant = std::variant<RefPtr<JSC::Uint8ClampedArray>, RefPtr<JSC::Float16Array>>;
    operator Variant() const;

    Ref<JSC::ArrayBufferView> m_arrayBufferView;
};

} // namespace WebCore
