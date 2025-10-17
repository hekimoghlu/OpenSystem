/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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

namespace API {

template<typename NumberType>
class Number {
public:
    NumberType value() const { return m_value; }
protected:
    explicit Number(NumberType value)
        : m_value(value) { }
private:
    const NumberType m_value;
};

class Boolean : public Number<bool>, public ObjectImpl<API::Object::Type::Boolean> {
public:
    static Ref<Boolean> create(bool value) { return adoptRef(*new Boolean(value)); }
private:
    explicit Boolean(bool value)
        : Number(value) { }
};

class Double : public Number<double>, public ObjectImpl<API::Object::Type::Double> {
public:
    static Ref<Double> create(double value) { return adoptRef(*new Double(value)); }
private:
    explicit Double(double value)
        : Number(value) { }
};

class UInt64 : public Number<uint64_t>, public ObjectImpl<API::Object::Type::UInt64> {
public:
    static Ref<UInt64> create(uint64_t value) { return adoptRef(*new UInt64(value)); }
private:
    explicit UInt64(uint64_t value)
        : Number(value) { }
};

class Int64 : public Number<int64_t>, public ObjectImpl<API::Object::Type::Int64> {
public:
    static Ref<Int64> create(int64_t value) { return adoptRef(*new Int64(value)); }
private:
    explicit Int64(int64_t value)
        : Number(value) { }
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(Boolean);
SPECIALIZE_TYPE_TRAITS_API_OBJECT(Double);
SPECIALIZE_TYPE_TRAITS_API_OBJECT(UInt64);
SPECIALIZE_TYPE_TRAITS_API_OBJECT(Int64);
