/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#include "config.h"
#include "Constraints.h"

#include "TypeStore.h"

namespace WGSL {

bool satisfies(const Type* type, Constraint constraint)
{
    if (constraint == Constraints::None)
        return true;

    auto* primitive = std::get_if<Types::Primitive>(type);
    if (!primitive) {
        if (auto* reference = std::get_if<Types::Reference>(type))
            return satisfies(reference->element, constraint);
        return false;
    }

    switch (primitive->kind) {
    case Types::Primitive::AbstractInt:
        return constraint >= Constraints::AbstractInt;
    case Types::Primitive::AbstractFloat:
        return constraint & Constraints::Float;
    case Types::Primitive::I32:
        return constraint & Constraints::I32;
    case Types::Primitive::U32:
        return constraint & Constraints::U32;
    case Types::Primitive::F32:
        return constraint & Constraints::F32;
    case Types::Primitive::F16:
        return constraint & Constraints::F16;
    case Types::Primitive::Bool:
        return constraint & Constraints::Bool;

    case Types::Primitive::Void:
    case Types::Primitive::Sampler:
    case Types::Primitive::SamplerComparison:
    case Types::Primitive::TextureExternal:
    case Types::Primitive::AccessMode:
    case Types::Primitive::TexelFormat:
    case Types::Primitive::AddressSpace:
        return false;
    }
}

const Type* satisfyOrPromote(const Type* type, Constraint constraint, const TypeStore& types)
{
    if (constraint == Constraints::None)
        return type;

    auto* primitive = std::get_if<Types::Primitive>(type);
    if (!primitive) {
        if (auto* reference = std::get_if<Types::Reference>(type))
            return satisfyOrPromote(reference->element, constraint, types);
        return nullptr;
    }

    switch (primitive->kind) {
    case Types::Primitive::AbstractInt:
        if (constraint < Constraints::AbstractInt)
            return nullptr;
        if (constraint & Constraints::AbstractInt)
            return type;
        if (constraint & Constraints::I32)
            return types.i32Type();
        if (constraint & Constraints::U32)
            return types.u32Type();
        if (constraint & Constraints::AbstractFloat)
            return types.abstractFloatType();
        if (constraint & Constraints::F32)
            return types.f32Type();
        if (constraint & Constraints::F16)
            return types.f16Type();
        RELEASE_ASSERT_NOT_REACHED();
    case Types::Primitive::AbstractFloat:
        if (constraint < Constraints::AbstractFloat)
            return nullptr;
        if (constraint & Constraints::AbstractFloat)
            return type;
        if (constraint & Constraints::F32)
            return types.f32Type();
        if (constraint & Constraints::F16)
            return types.f16Type();
        RELEASE_ASSERT_NOT_REACHED();
    case Types::Primitive::I32:
        if (!(constraint & Constraints::I32))
            return nullptr;
        return type;
    case Types::Primitive::U32:
        if (!(constraint & Constraints::U32))
            return nullptr;
        return type;
    case Types::Primitive::F32:
        if (!(constraint & Constraints::F32))
            return nullptr;
        return type;
    case Types::Primitive::F16:
        if (!(constraint & Constraints::F16))
            return nullptr;
        return type;
    case Types::Primitive::Bool:
        if (!(constraint & Constraints::Bool))
            return nullptr;
        return type;

    case Types::Primitive::Void:
    case Types::Primitive::Sampler:
    case Types::Primitive::SamplerComparison:
    case Types::Primitive::TextureExternal:
    case Types::Primitive::AccessMode:
    case Types::Primitive::TexelFormat:
    case Types::Primitive::AddressSpace:
        return nullptr;
    }
}

const Type* concretize(const Type* type, TypeStore& types)
{
    using namespace Types;

    return WTF::switchOn(*type,
        [&](const Primitive&) -> const Type* {
            return satisfyOrPromote(type, Constraints::ConcreteScalar, types);
        },
        [&](const Vector& vector) -> const Type* {
            auto* element = concretize(vector.element, types);
            if (!element)
                return nullptr;
            return types.vectorType(vector.size, element);
        },
        [&](const Matrix& matrix) -> const Type* {
            auto* element = concretize(matrix.element, types);
            if (!element)
                return nullptr;
            return types.matrixType(matrix.columns, matrix.rows, element);
        },
        [&](const Array& array) -> const Type* {
            auto* element = concretize(array.element, types);
            if (!element)
                return nullptr;
            return types.arrayType(element, array.size);
        },
        [&](const Struct&) -> const Type* {
            return type;
        },
        [&](const PrimitiveStruct& primitiveStruct) -> const Type* {
            switch (primitiveStruct.kind) {
            case PrimitiveStruct::FrexpResult::kind: {
                auto* fract = concretize(primitiveStruct.values[PrimitiveStruct::FrexpResult::fract], types);
                auto* exp = concretize(primitiveStruct.values[PrimitiveStruct::FrexpResult::exp], types);
                if (!fract || !exp)
                    return nullptr;
                return types.frexpResultType(fract, exp);
            }
            case PrimitiveStruct::ModfResult::kind: {
                auto* fract = concretize(primitiveStruct.values[PrimitiveStruct::ModfResult::fract], types);
                auto* whole = concretize(primitiveStruct.values[PrimitiveStruct::ModfResult::whole], types);
                if (!fract || !whole)
                    return nullptr;
                return types.modfResultType(fract, whole);
            }
            case PrimitiveStruct::AtomicCompareExchangeResult::kind: {
                return type;
            }
            }
        },
        [&](const Pointer&) -> const Type* {
            return type;
        },
        [&](const Bottom&) -> const Type* {
            return type;
        },
        [&](const Atomic&) -> const Type* {
            return type;
        },
        [&](const Function&) -> const Type* {
            return nullptr;
        },
        [&](const Texture&) -> const Type* {
            return nullptr;
        },
        [&](const TextureStorage&) -> const Type* {
            return nullptr;
        },
        [&](const TextureDepth&) -> const Type* {
            return nullptr;
        },
        [&](const Reference&) -> const Type* {
            return nullptr;
        },
        [&](const TypeConstructor&) -> const Type* {
            return nullptr;
        });
}

} // namespace WGSL
