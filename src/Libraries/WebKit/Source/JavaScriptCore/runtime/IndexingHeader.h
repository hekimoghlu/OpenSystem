/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "PropertyStorage.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class ArrayBuffer;
class Butterfly;
class LLIntOffsetsExtractor;
class Structure;
struct ArrayStorage;

class IndexingHeader {
public:
    // Define the maximum storage vector length to be 2^32 / sizeof(JSValue) / 2 to ensure that
    // there is no risk of overflow.
    static constexpr unsigned maximumLength = 0x10000000;
    
    static constexpr ptrdiff_t offsetOfIndexingHeader() { return -static_cast<ptrdiff_t>(sizeof(IndexingHeader)); }
    
    static constexpr ptrdiff_t offsetOfArrayBuffer() { return OBJECT_OFFSETOF(IndexingHeader, u.typedArray.buffer); }
    static constexpr ptrdiff_t offsetOfPublicLength() { return OBJECT_OFFSETOF(IndexingHeader, u.lengths.publicLength); }
    static constexpr ptrdiff_t offsetOfVectorLength() { return OBJECT_OFFSETOF(IndexingHeader, u.lengths.vectorLength); }
    
    IndexingHeader()
    {
        u.lengths.publicLength = 0;
        u.lengths.vectorLength = 0;
    }
    
    uint32_t vectorLength() const { return u.lengths.vectorLength; }
    
    void setVectorLength(uint32_t length)
    {
        RELEASE_ASSERT(length <= maximumLength);
        u.lengths.vectorLength = length;
    }
    
    uint32_t publicLength() const { return u.lengths.publicLength; }
    void setPublicLength(uint32_t auxWord) { u.lengths.publicLength = auxWord; }
    
    ArrayBuffer* arrayBuffer() { return u.typedArray.buffer; }
    void setArrayBuffer(ArrayBuffer* buffer) { u.typedArray.buffer = buffer; }
    
    static IndexingHeader* from(Butterfly* butterfly)
    {
        return reinterpret_cast<IndexingHeader*>(butterfly) - 1;
    }
    
    static const IndexingHeader* from(const Butterfly* butterfly)
    {
        return reinterpret_cast<const IndexingHeader*>(butterfly) - 1;
    }
    
    static IndexingHeader* from(ArrayStorage* arrayStorage)
    {
        return const_cast<IndexingHeader*>(from(const_cast<const ArrayStorage*>(arrayStorage)));
    }
    
    static const IndexingHeader* from(const ArrayStorage* arrayStorage)
    {
        return reinterpret_cast<const IndexingHeader*>(arrayStorage) - 1;
    }
    
    static IndexingHeader* fromEndOf(PropertyStorage propertyStorage)
    {
        return reinterpret_cast<IndexingHeader*>(propertyStorage);
    }
    
    PropertyStorage propertyStorage()
    {
        return reinterpret_cast_ptr<PropertyStorage>(this);
    }
    
    ConstPropertyStorage propertyStorage() const
    {
        return reinterpret_cast_ptr<ConstPropertyStorage>(this);
    }
    
    ArrayStorage* arrayStorage()
    {
        return reinterpret_cast<ArrayStorage*>(this + 1);
    }
    
    Butterfly* butterfly()
    {
        return reinterpret_cast<Butterfly*>(this + 1);
    }
    
    // These methods are not standalone in the sense that they cannot be
    // used on a copy of the IndexingHeader.
    size_t preCapacity(Structure*);
    size_t indexingPayloadSizeInBytes(Structure*);
    
private:
    friend class LLIntOffsetsExtractor;

    union {
        struct {
            // FIXME: vectorLength should be least significant, so that it's really hard to craft a pointer by
            // mucking with the butterfly.
            // https://bugs.webkit.org/show_bug.cgi?id=174927
            uint32_t publicLength; // The meaning of this field depends on the array type, but for all JSArrays we rely on this being the publicly visible length (array.length).
            uint32_t vectorLength; // The length of the indexed property storage. The actual size of the storage depends on this, and the type.
        } lengths;
        
        struct {
            ArrayBuffer* buffer;
        } typedArray;
    } u;
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
