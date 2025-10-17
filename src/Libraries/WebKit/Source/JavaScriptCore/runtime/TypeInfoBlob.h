/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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

#include "CellState.h"
#include "IndexingType.h"
#include "JSTypeInfo.h"
#include "StructureID.h"

namespace JSC {

class TypeInfoBlob {
    friend class LLIntOffsetsExtractor;
public:
    TypeInfoBlob() = default;

    TypeInfoBlob(IndexingType indexingModeIncludingHistory, const TypeInfo& typeInfo)
    {
        u.fields.indexingModeIncludingHistory = indexingModeIncludingHistory;
        u.fields.type = typeInfo.type();
        u.fields.inlineTypeFlags = typeInfo.inlineTypeFlags();
        u.fields.defaultCellState = CellState::DefinitelyWhite;
    }

    void operator=(const TypeInfoBlob& other) { u.word = other.u.word; }

    IndexingType indexingModeIncludingHistory() const { return u.fields.indexingModeIncludingHistory; }
    Dependency fencedIndexingModeIncludingHistory(IndexingType& indexingType)
    {
        return Dependency::loadAndFence(&u.fields.indexingModeIncludingHistory, indexingType);
    }
    void setIndexingModeIncludingHistory(IndexingType indexingModeIncludingHistory) { u.fields.indexingModeIncludingHistory = indexingModeIncludingHistory; }
    JSType type() const { return u.fields.type; }
    TypeInfo::InlineTypeFlags inlineTypeFlags() const { return u.fields.inlineTypeFlags; }
    
    TypeInfo typeInfo(TypeInfo::OutOfLineTypeFlags outOfLineTypeFlags) const { return TypeInfo(type(), inlineTypeFlags(), outOfLineTypeFlags); }
    CellState defaultCellState() const { return u.fields.defaultCellState; }

    static constexpr int32_t typeInfoBlob(IndexingType indexingModeIncludingHistory, JSType type, TypeInfo::InlineTypeFlags inlineTypeFlags)
    {
#if CPU(LITTLE_ENDIAN)
        return static_cast<int32_t>((static_cast<uint32_t>(indexingModeIncludingHistory) << 0) | (static_cast<uint32_t>(type) << 8) | (static_cast<uint32_t>(inlineTypeFlags) << 16) | (static_cast<uint32_t>(CellState::DefinitelyWhite) << 24));
#else
        return static_cast<int32_t>((static_cast<uint32_t>(indexingModeIncludingHistory) << 24) | (static_cast<uint32_t>(type) << 16) | (static_cast<uint32_t>(inlineTypeFlags) << 8) | (static_cast<uint32_t>(CellState::DefinitelyWhite) << 0));
#endif
    }

    int32_t blob() const { return u.word; }

    static constexpr ptrdiff_t indexingModeIncludingHistoryOffset()
    {
        return OBJECT_OFFSETOF(TypeInfoBlob, u.fields.indexingModeIncludingHistory);
    }

private:
    union Data {
        struct {
            IndexingType indexingModeIncludingHistory;
            JSType type;
            TypeInfo::InlineTypeFlags inlineTypeFlags;
            CellState defaultCellState;
        } fields;
        int32_t word;

        Data() { word = 0xbbadbeef; }
    };

    Data u;
};

} // namespace JSC
