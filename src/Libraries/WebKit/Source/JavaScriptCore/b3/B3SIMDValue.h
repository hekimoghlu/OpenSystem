/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 18, 2021.
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

#if ENABLE(B3_JIT)

#include "B3Value.h"
#include "SIMDInfo.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class JS_EXPORT_PRIVATE SIMDValue final : public Value {
public:
    static bool accepts(Kind kind)
    {
        switch (kind.opcode()) {
        case VectorExtractLane:
        case VectorReplaceLane:
        case VectorDupElement:
        case VectorEqual:
        case VectorNotEqual:
        case VectorLessThan:
        case VectorLessThanOrEqual:
        case VectorBelow:
        case VectorBelowOrEqual:
        case VectorGreaterThan:
        case VectorGreaterThanOrEqual:
        case VectorAbove:
        case VectorAboveOrEqual:
        case VectorAdd:
        case VectorSub:
        case VectorAddSat:
        case VectorSubSat:
        case VectorMul:
        case VectorDiv:
        case VectorMin:
        case VectorMax:
        case VectorPmin:
        case VectorPmax:
        case VectorNarrow:
        case VectorNot:
        case VectorAnd:
        case VectorAndnot:
        case VectorOr:
        case VectorXor:
        case VectorShl:
        case VectorShr:
        case VectorAbs:
        case VectorNeg:
        case VectorPopcnt:
        case VectorCeil:
        case VectorFloor:
        case VectorTrunc:
        case VectorTruncSat:
        case VectorRelaxedTruncSat:
        case VectorConvert:
        case VectorConvertLow:
        case VectorNearest:
        case VectorSqrt:
        case VectorExtendLow:
        case VectorExtendHigh:
        case VectorPromote:
        case VectorDemote:
        case VectorSplat:
        case VectorAnyTrue:
        case VectorAllTrue:
        case VectorAvgRound:
        case VectorBitmask:
        case VectorBitwiseSelect:
        case VectorExtaddPairwise:
        case VectorMulSat:
        case VectorSwizzle:
        case VectorMulByElement:
        case VectorShiftByVector:
        case VectorDotProduct:
        case VectorRelaxedSwizzle:
        case VectorRelaxedMAdd:
        case VectorRelaxedNMAdd:
        case VectorRelaxedLaneSelect:
            return true;
        default:
            return false;
        }
    }

    ~SIMDValue() final;

    SIMDInfo simdInfo() const { return m_simdInfo; }
    SIMDLane simdLane() const { return m_simdInfo.lane; }
    SIMDSignMode signMode() const { return m_simdInfo.signMode; }
    uint8_t immediate() const { return m_immediate; }

    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_FIXED_CHILDREN

protected:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    template<typename... Arguments>
    SIMDValue(Origin origin, Kind kind, Type type, SIMDInfo info, uint8_t imm1, Arguments... arguments)
        : Value(CheckedOpcode, kind, type, static_cast<NumChildren>(sizeof...(arguments)), origin, arguments...)
        , m_simdInfo(info)
        , m_immediate(imm1)
    {
    }

    template<typename... Arguments>
    SIMDValue(Origin origin, Kind kind, Type type, SIMDInfo info, Arguments... arguments)
        : Value(CheckedOpcode, kind, type, static_cast<NumChildren>(sizeof...(arguments)), origin, arguments...)
        , m_simdInfo(info)
    {
    }

    template<typename... Arguments>
    SIMDValue(Origin origin, Kind kind, Type type, SIMDLane simdLane, SIMDSignMode signMode, uint8_t imm1, Arguments... arguments)
        : Value(CheckedOpcode, kind, type, static_cast<NumChildren>(sizeof...(arguments)), origin, arguments...)
        , m_simdInfo { simdLane, signMode }
        , m_immediate(imm1)
    {
    }

    template<typename... Arguments>
    SIMDValue(Origin origin, Kind kind, Type type, SIMDLane simdLane, SIMDSignMode signMode, Arguments... arguments)
        : Value(CheckedOpcode, kind, type, static_cast<NumChildren>(sizeof...(arguments)), origin, arguments...)
        , m_simdInfo { simdLane, signMode }
    {
    }

private:
    template<typename... Arguments>
    static Opcode opcodeFromConstructor(Origin, Kind kind, Type, SIMDLane, SIMDSignMode, Arguments...) { return kind.opcode(); }
    template<typename... Arguments>
    static Opcode opcodeFromConstructor(Origin, Kind kind, Type, SIMDInfo, Arguments...) { return kind.opcode(); }
    template<typename... Arguments>
    static Opcode opcodeFromConstructor(Origin, Kind kind, Type, SIMDLane, SIMDSignMode, v128_t, Arguments...) { return kind.opcode(); }
    friend class Procedure;
    friend class Value;

    SIMDInfo m_simdInfo { };
    uint8_t m_immediate { 0 };
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
