/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 13, 2024.
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

namespace JSC {

enum class GetByIdMode : uint8_t {
    ProtoLoad = 0, // This must be zero to reuse the higher bits of the pointer as this ProtoLoad mode.
    Default = 1,
    Unset = 2,
    ArrayLength = 3,
};

struct GetByIdModeMetadataDefault {
    StructureID structureID;
    PropertyOffset cachedOffset;
    unsigned padding1;
};
static_assert(sizeof(GetByIdModeMetadataDefault) == 12);

struct GetByIdModeMetadataUnset {
    StructureID structureID;
    unsigned padding1;
    unsigned padding2;
};
static_assert(sizeof(GetByIdModeMetadataUnset) == 12);

struct GetByIdModeMetadataArrayLength {
    unsigned padding1;
    unsigned padding2;
    unsigned padding3;
};
static_assert(sizeof(GetByIdModeMetadataArrayLength) == 12);

struct GetByIdModeMetadataProtoLoad {
    StructureID structureID;
    PropertyOffset cachedOffset;
    JSObject* cachedSlot;
};
#if CPU(LITTLE_ENDIAN) && CPU(ADDRESS64)
static_assert(sizeof(GetByIdModeMetadataProtoLoad) == 16);
#endif

// In 64bit Little endian architecture, this union shares ProtoLoad's JSObject* cachedSlot with "hitCountForLLIntCaching" and "mode".
// This is possible because these values must be zero if we use ProtoLoad mode.
#if CPU(LITTLE_ENDIAN) && CPU(ADDRESS64)
union GetByIdModeMetadata {
    GetByIdModeMetadata()
    {
        defaultMode.structureID = StructureID();
        defaultMode.cachedOffset = 0;
        defaultMode.padding1 = 0;
        mode = GetByIdMode::Default;
        hitCountForLLIntCaching = Options::prototypeHitCountForLLIntCaching();
    }

    void clearToDefaultModeWithoutCache();
    void setUnsetMode(Structure*);
    void setArrayLengthMode();
    void setProtoLoadMode(Structure*, PropertyOffset, JSObject*);

    struct {
        uint32_t padding1;
        uint32_t padding2;
        uint32_t padding3;
        uint16_t padding4;
        GetByIdMode mode;
        uint8_t hitCountForLLIntCaching; // This must be zero when we use ProtoLoad mode.
    };
    static constexpr ptrdiff_t offsetOfMode() { return OBJECT_OFFSETOF(GetByIdModeMetadata, mode); }
    GetByIdModeMetadataDefault defaultMode;
    GetByIdModeMetadataUnset unsetMode;
    GetByIdModeMetadataArrayLength arrayLengthMode;
    GetByIdModeMetadataProtoLoad protoLoadMode;
};
static_assert(sizeof(GetByIdModeMetadata) == 16);
#else
struct GetByIdModeMetadata {
    GetByIdModeMetadata()
    {
        defaultMode.structureID = StructureID();
        defaultMode.cachedOffset = 0;
        defaultMode.padding1 = 0;
        mode = GetByIdMode::Default;
        hitCountForLLIntCaching = Options::prototypeHitCountForLLIntCaching();
    }

    void clearToDefaultModeWithoutCache();
    void setUnsetMode(Structure*);
    void setArrayLengthMode();
    void setProtoLoadMode(Structure*, PropertyOffset, JSObject*);

    union {
        GetByIdModeMetadataDefault defaultMode;
        GetByIdModeMetadataUnset unsetMode;
        GetByIdModeMetadataArrayLength arrayLengthMode;
        GetByIdModeMetadataProtoLoad protoLoadMode;
    };
    GetByIdMode mode;
    static constexpr ptrdiff_t offsetOfMode() { return OBJECT_OFFSETOF(GetByIdModeMetadata, mode); }
    uint8_t hitCountForLLIntCaching;
};
#endif

inline void GetByIdModeMetadata::clearToDefaultModeWithoutCache()
{
    mode = GetByIdMode::Default;
    defaultMode.structureID = StructureID();
    defaultMode.cachedOffset = 0;
}

inline void GetByIdModeMetadata::setUnsetMode(Structure* structure)
{
    mode = GetByIdMode::Unset;
    unsetMode.structureID = structure->id();
    defaultMode.cachedOffset = 0;
}

inline void GetByIdModeMetadata::setArrayLengthMode()
{
    mode = GetByIdMode::ArrayLength;
    // We should clear the structure ID to avoid the old structure ID being saved.
    defaultMode.structureID = StructureID();
    defaultMode.cachedOffset = 0;
    // Prevent the prototype cache from ever happening.
    hitCountForLLIntCaching = 0;
}

inline void GetByIdModeMetadata::setProtoLoadMode(Structure* structure, PropertyOffset offset, JSObject* cachedSlot)
{
#if CPU(LITTLE_ENDIAN) && CPU(ADDRESS64)
    // We rely on ProtoLoad being 0, or else the high bits of the pointer would write the wrong mode and hit count
    static_assert(!static_cast<std::underlying_type_t<GetByIdMode>>(GetByIdMode::ProtoLoad)); // In 64bit architecture, this field is shared with protoLoadMode.cachedSlot.
#else
    mode = GetByIdMode::ProtoLoad;
#endif

    protoLoadMode.structureID = structure->id();
    protoLoadMode.cachedOffset = offset;

    // We know that this pointer will remain valid because it will be cleared by either a watchpoint fire or
    // during GC when we clear the LLInt caches.

    // The write to cachedSlot also writes the mode, since they overlap in the struct layout. We know that
    // the mode ProtoLoad is 0 by the static assertion above.
    protoLoadMode.cachedSlot = cachedSlot;

    ASSERT(mode == GetByIdMode::ProtoLoad);
    ASSERT(!hitCountForLLIntCaching);
    ASSERT(protoLoadMode.structureID == structure->id());
    ASSERT(protoLoadMode.cachedOffset == offset);
    ASSERT(protoLoadMode.cachedSlot == cachedSlot);
}

} // namespace JSC
