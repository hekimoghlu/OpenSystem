/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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

#include "Types.h"
#include <wtf/FixedVector.h>
#include <wtf/HashMap.h>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>

namespace WGSL {

struct Type;

namespace AST {
class Identifier;
}

class TypeCache {
public:
    enum KeyKind : uint8_t {
        Vector = 1,
        Matrix,
        Array,
        Texture,
        TextureStorage,
        Reference,
        Pointer,
        PrimitiveStruct,
    };

    using EncodedKey = std::tuple<uint8_t, uint8_t, uint16_t, uint32_t, uintptr_t>;

    template<typename Key>
    const Type* find(const Key&) const;

    template<typename Key>
    void insert(const Key&, const Type*);

private:
    HashMap<EncodedKey, const Type*> m_storage;
};

class TypeStore {
public:
    TypeStore();

    const Type* bottomType() const { return m_bottom; }
    const Type* voidType() const { return m_void; }
    const Type* boolType() const { return m_bool; }

    const Type* abstractIntType() const { return m_abstractInt; }
    const Type* i32Type() const { return m_i32; }
    const Type* u32Type() const { return m_u32; }

    const Type* abstractFloatType() const { return m_abstractFloat; }
    const Type* f32Type() const { return m_f32; }
    const Type* f16Type() const { return m_f16; }
    const Type* samplerType() const { return m_sampler; }
    const Type* samplerComparisonType() const { return m_samplerComparison; }
    const Type* textureExternalType() const { return m_textureExternal; }
    const Type* accessModeType() const { return m_accessMode; }
    const Type* texelFormatType() const { return m_texelFormat; }
    const Type* addressSpaceType() const { return m_addressSpace; }

    const Type* textureDepth2dType() const { return m_textureDepth2d; }
    const Type* textureDepth2dArrayType() const { return m_textureDepthArray2d; }
    const Type* textureDepthCubeType() const { return m_textureDepthCube; }
    const Type* textureDepthCubeArrayType() const { return m_textureDepthArrayCube; }
    const Type* textureDepthMultisampled2dType() const { return m_textureDepthMultisampled2d; }

    const Type* structType(AST::Structure&, HashMap<String, const Type*>&& = { });
    const Type* arrayType(const Type*, Types::Array::Size);
    const Type* vectorType(uint8_t, const Type*);
    const Type* matrixType(uint8_t columns, uint8_t rows, const Type*);
    const Type* textureType(Types::Texture::Kind, const Type*);
    const Type* textureStorageType(Types::TextureStorage::Kind, TexelFormat, AccessMode);
    const Type* functionType(Vector<const Type*>&&, const Type*, bool mustUse);
    const Type* referenceType(AddressSpace, const Type*, AccessMode, bool isVectorComponent = false);
    const Type* pointerType(AddressSpace, const Type*, AccessMode);
    const Type* atomicType(const Type*);
    const Type* typeConstructorType(ASCIILiteral, std::function<const Type*(AST::ElaboratedTypeExpression&)>&&);
    const Type* frexpResultType(const Type*, const Type*);
    const Type* modfResultType(const Type*, const Type*);
    const Type* atomicCompareExchangeResultType(const Type*);

private:
    template<typename TypeKind, typename... Arguments>
    const Type* allocateType(Arguments&&...);

    Vector<std::unique_ptr<const Type>> m_types;
    TypeCache m_cache;

    const Type* m_bottom;
    const Type* m_abstractInt;
    const Type* m_abstractFloat;
    const Type* m_void;
    const Type* m_bool;
    const Type* m_i32;
    const Type* m_u32;
    const Type* m_f32;
    const Type* m_f16;
    const Type* m_sampler;
    const Type* m_samplerComparison;
    const Type* m_textureExternal;
    const Type* m_accessMode;
    const Type* m_texelFormat;
    const Type* m_addressSpace;
    const Type* m_textureDepth2d;
    const Type* m_textureDepthArray2d;
    const Type* m_textureDepthCube;
    const Type* m_textureDepthArrayCube;
    const Type* m_textureDepthMultisampled2d;
    const Type* m_atomicI32;
    const Type* m_atomicU32;
    const Type* m_atomicCompareExchangeResultI32 { nullptr };
    const Type* m_atomicCompareExchangeResultU32 { nullptr };
};

} // namespace WGSL
