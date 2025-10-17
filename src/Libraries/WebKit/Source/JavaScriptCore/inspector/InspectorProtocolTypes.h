/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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

#include <wtf/Assertions.h>
#include <wtf/Expected.h>
#include <wtf/JSONValues.h>
#include <wtf/text/WTFString.h>

namespace Inspector {

namespace Protocol {

using ErrorString = String;

template <typename T>
using ErrorStringOr = Expected<T, ErrorString>;

template<typename> struct BindingTraits;

template<JSON::Value::Type type> struct PrimitiveBindingTraits {
    static void assertValueHasExpectedType(JSON::Value* value)
    {
        ASSERT_UNUSED(value, value);
        ASSERT_UNUSED(value, value->type() == type);
    }
};

template<typename T> struct BindingTraits<JSON::ArrayOf<T>> {
    static Ref<JSON::ArrayOf<T>> runtimeCast(Ref<JSON::Value>&& value)
    {
        auto array = value->asArray();
        BindingTraits<JSON::ArrayOf<T>>::assertValueHasExpectedType(array.get());
        static_assert(sizeof(JSON::ArrayOf<T>) == sizeof(JSON::Array), "type cast problem");
        return static_reference_cast<JSON::ArrayOf<T>>(static_reference_cast<JSON::ArrayBase>(array.releaseNonNull()));
    }

    static void assertValueHasExpectedType(JSON::Value* value)
    {
        ASSERT_UNUSED(value, value);
#if ASSERT_ENABLED
        auto array = value->asArray();
        ASSERT(array);
        for (unsigned i = 0; i < array->length(); i++)
            BindingTraits<T>::assertValueHasExpectedType(array->get(i).ptr());
#endif
    }
};

template<> struct BindingTraits<JSON::Value> {
    static void assertValueHasExpectedType(JSON::Value*)
    {
    }
};

template<> struct BindingTraits<JSON::Array> : PrimitiveBindingTraits<JSON::Value::Type::Array> { };
template<> struct BindingTraits<JSON::Object> : PrimitiveBindingTraits<JSON::Value::Type::Object> { };
template<> struct BindingTraits<String> : PrimitiveBindingTraits<JSON::Value::Type::String> { };
template<> struct BindingTraits<bool> : PrimitiveBindingTraits<JSON::Value::Type::Boolean> { };
template<> struct BindingTraits<double> : PrimitiveBindingTraits<JSON::Value::Type::Double> { };
template<> struct BindingTraits<int> : PrimitiveBindingTraits<JSON::Value::Type::Integer> { };

}

} // namespace Inspector
