/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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

#include "Constraints.h"
#include "TypeStore.h"
#include "WGSL.h"

namespace WGSL {

struct ValueVariable {
    unsigned id;
};
using AbstractValue = std::variant<ValueVariable, unsigned>;

struct TypeVariable {
    unsigned id;
    Constraint constraints { Constraints::None };
};

struct AbstractVector;
struct AbstractMatrix;
struct AbstractTexture;
struct AbstractTextureStorage;
struct AbstractChannelFormat;
struct AbstractReference;
struct AbstractPointer;
struct AbstractArray;
struct AbstractAtomic;

using AbstractTypeImpl = std::variant<
    AbstractVector,
    AbstractMatrix,
    AbstractTexture,
    AbstractTextureStorage,
    AbstractChannelFormat,
    AbstractReference,
    AbstractPointer,
    AbstractArray,
    AbstractAtomic,
    TypeVariable,
    const Type*
>;
using AbstractType = std::unique_ptr<AbstractTypeImpl>;

struct AbstractVector {
    AbstractValue size;
    AbstractType element;
};

struct AbstractMatrix {
    AbstractValue columns;
    AbstractValue rows;
    AbstractType element;
};

struct AbstractTexture {
    Types::Texture::Kind kind;
    AbstractType element;
};

struct AbstractTextureStorage {
    Types::TextureStorage::Kind kind;
    AbstractValue format;
    AbstractValue access;
};

struct AbstractChannelFormat {
    AbstractValue format;
};

struct AbstractReference {
    AbstractValue addressSpace;
    AbstractType element;
    AbstractValue accessMode;
};

struct AbstractPointer {
    AbstractValue addressSpace;
    AbstractType element;
    AbstractValue accessMode;

    AbstractPointer(AbstractValue, AbstractType);
    AbstractPointer(AbstractValue, AbstractType, AbstractValue);
};

struct AbstractArray {
    AbstractType element;
};

struct AbstractAtomic {
    AbstractType element;
};

struct OverloadCandidate {
    Vector<TypeVariable, 1> typeVariables;
    Vector<ValueVariable, 2> valueVariables;
    Vector<AbstractType, 2> parameters;
    AbstractType result;
};

struct OverloadedDeclaration {
    enum Kind : uint8_t {
        Operator,
        Constructor,
        Function,
    };

    Kind kind;
    bool mustUse;

    Expected<ConstantValue, String> (*constantFunction)(const Type*, const FixedVector<ConstantValue>&);
    OptionSet<ShaderStage> visibility;
    Vector<OverloadCandidate> overloads;
};

struct SelectedOverload {
    FixedVector<const Type*> parameters;
    const Type* result;
};

std::optional<SelectedOverload> resolveOverloads(TypeStore&, const Vector<OverloadCandidate>&, const Vector<const Type*>& valueArguments, const Vector<const Type*>& typeArguments);

template<typename T>
static AbstractType allocateAbstractType(const T& type)
{
    return std::unique_ptr<AbstractTypeImpl>(new AbstractTypeImpl(type));
}

template<typename T>
static AbstractType allocateAbstractType(T&& type)
{
    return std::unique_ptr<AbstractTypeImpl>(new AbstractTypeImpl(WTFMove(type)));
}

} // namespace WGSL

namespace WTF {
void printInternal(PrintStream&, const WGSL::ValueVariable&);
void printInternal(PrintStream&, const WGSL::AbstractValue&);
void printInternal(PrintStream&, const WGSL::TypeVariable&);
void printInternal(PrintStream&, const WGSL::AbstractType&);
void printInternal(PrintStream&, const WGSL::OverloadCandidate&);
void printInternal(PrintStream&, WGSL::Types::Texture::Kind);
void printInternal(PrintStream&, WGSL::Types::TextureStorage::Kind);
} // namespace WTF
