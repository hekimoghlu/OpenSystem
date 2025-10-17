/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#include "BasicBlockLocation.h"
#include "SourceID.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class VM;

struct BasicBlockKey {
    BasicBlockKey()
        : m_startOffset(-3)
        , m_endOffset(-3)
    { }

    BasicBlockKey(int startOffset, int endOffset)
        : m_startOffset(startOffset)
        , m_endOffset(endOffset)
    { }

    BasicBlockKey(WTF::HashTableDeletedValueType)
        : m_startOffset(-2)
        , m_endOffset(-2)
    { }

    bool isHashTableDeletedValue() const { return m_startOffset == -2 && m_endOffset == -2; }
    friend bool operator==(const BasicBlockKey&, const BasicBlockKey&) = default;
    unsigned hash() const { return m_startOffset + m_endOffset + 1; }

    int m_startOffset;
    int m_endOffset;
};

struct BasicBlockKeyHash {
    static unsigned hash(const BasicBlockKey& key) { return key.hash(); }
    static bool equal(const BasicBlockKey& a, const BasicBlockKey& b) { return a == b; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

} // namespace JSC

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<JSC::BasicBlockKey> : JSC::BasicBlockKeyHash { };

template<typename T> struct HashTraits;
template<> struct HashTraits<JSC::BasicBlockKey> : SimpleClassHashTraits<JSC::BasicBlockKey> {
    static constexpr bool emptyValueIsZero = false;
};

} // namespace WTF

namespace JSC {

struct BasicBlockRange {
    int m_startOffset;
    int m_endOffset;
    bool m_hasExecuted;
    size_t m_executionCount;
};

class ControlFlowProfiler {
    WTF_MAKE_TZONE_ALLOCATED(ControlFlowProfiler);
public:
    ControlFlowProfiler();
    ~ControlFlowProfiler();
    BasicBlockLocation* getBasicBlockLocation(SourceID, int startOffset, int endOffset);
    JS_EXPORT_PRIVATE void dumpData() const;
    Vector<BasicBlockRange> getBasicBlocksForSourceID(SourceID, VM&) const;
    BasicBlockLocation* dummyBasicBlock() { return &m_dummyBasicBlock; }
    JS_EXPORT_PRIVATE bool hasBasicBlockAtTextOffsetBeenExecuted(int, SourceID, VM&); // This function exists for testing.
    JS_EXPORT_PRIVATE size_t basicBlockExecutionCountAtTextOffset(int, SourceID, VM&); // This function exists for testing.

private:
    typedef UncheckedKeyHashMap<BasicBlockKey, BasicBlockLocation*> BlockLocationCache;
    typedef UncheckedKeyHashMap<SourceID, BlockLocationCache> SourceIDBuckets;

    SourceIDBuckets m_sourceIDBuckets;
    BasicBlockLocation m_dummyBasicBlock;
};

} // namespace JSC
