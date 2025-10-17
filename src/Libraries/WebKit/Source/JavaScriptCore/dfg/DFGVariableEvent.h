/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 18, 2022.
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

#if ENABLE(DFG_JIT)

#include "DFGMinifiedID.h"
#include "DataFormat.h"
#include "MacroAssembler.h"
#include "Operands.h"
#include "VirtualRegister.h"
#include <stdio.h>

namespace JSC { namespace DFG {

enum VariableEventKind : uint8_t {
    // Marks the beginning of a checkpoint. If you interpret the variable
    // events starting at a Reset point then you'll get everything you need.
    Reset,
    
    // Node births. Points in the code where a node becomes relevant for OSR.
    // It may be the point where it is actually born (i.e. assigned) or it may
    // be a later point, if it's only later in the sequence of instructions
    // that we start to care about this node.
    BirthToFill,
    BirthToSpill,
    Birth,
    
    // Events related to how a node is represented.
    Fill,
    Spill,
    
    // Death of a node - after this we no longer care about this node.
    Death,
    
    // A MovHintEvent means that a node is being associated with a bytecode operand,
    // but that it has not been stored into that operand.
    MovHintEvent,
    
    // A SetLocalEvent means that a node's value has been stored into the stack.
    SetLocalEvent,
    
    // Used to indicate an uninitialized VariableEvent. Don't use for other
    // purposes.
    InvalidEventKind
};

union VariableRepresentation {
    VariableRepresentation() 
        : operand() 
    { }

    MacroAssembler::RegisterID gpr;
    MacroAssembler::FPRegisterID fpr;
#if USE(JSVALUE32_64)
    struct {
        MacroAssembler::RegisterID tagGPR;
        MacroAssembler::RegisterID payloadGPR;
    } pair;
#endif
    Operand operand;
};

class VariableEvent {
public:
    VariableEvent()
        : m_kind(InvalidEventKind)
    {
    }
    
    static VariableEvent reset()
    {
        VariableEvent event;
        event.m_kind = Reset;
        return event;
    }
    
    static VariableEvent fillGPR(VariableEventKind kind, MinifiedID id, MacroAssembler::RegisterID gpr, DataFormat dataFormat)
    {
        ASSERT(kind == BirthToFill || kind == Fill);
        ASSERT(dataFormat != DataFormatDouble);
#if USE(JSVALUE32_64)
        ASSERT(!(dataFormat & DataFormatJS));
#endif
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        VariableRepresentation representation;
        representation.gpr = gpr;
        event.m_kind = kind;
        event.m_dataFormat = dataFormat;
        event.m_which = WTFMove(which);
        event.m_representation = WTFMove(representation);
        return event;
    }
    
#if USE(JSVALUE32_64)
    static VariableEvent fillPair(VariableEventKind kind, MinifiedID id, MacroAssembler::RegisterID tagGPR, MacroAssembler::RegisterID payloadGPR)
    {
        ASSERT(kind == BirthToFill || kind == Fill);
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        VariableRepresentation representation;
        representation.pair.tagGPR = tagGPR;
        representation.pair.payloadGPR = payloadGPR;
        event.m_kind = kind;
        event.m_dataFormat = DataFormatJS;
        event.m_which = WTFMove(which);
        event.m_representation = WTFMove(representation);
        return event;
    }
#endif // USE(JSVALUE32_64)
    
    static VariableEvent fillFPR(VariableEventKind kind, MinifiedID id, MacroAssembler::FPRegisterID fpr)
    {
        ASSERT(kind == BirthToFill || kind == Fill);
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        VariableRepresentation representation;
        representation.fpr = fpr;
        event.m_kind = kind;
        event.m_dataFormat = DataFormatDouble;
        event.m_which = WTFMove(which);
        event.m_representation = WTFMove(representation);
        return event;
    }
    
    static VariableEvent birth(MinifiedID id)
    {
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        event.m_kind = Birth;
        event.m_which = WTFMove(which);
        return event;
    }
    
    static VariableEvent spill(VariableEventKind kind, MinifiedID id, VirtualRegister virtualRegister, DataFormat format)
    {
        ASSERT(kind == BirthToSpill || kind == Spill);
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        VariableRepresentation representation;
        representation.operand = virtualRegister;
        event.m_kind = kind;
        event.m_dataFormat = format;
        event.m_which = WTFMove(which);
        event.m_representation = WTFMove(representation);
        return event;
    }
    
    static VariableEvent death(MinifiedID id)
    {
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        event.m_kind = Death;
        event.m_which = WTFMove(which);
        return event;
    }
    
    static VariableEvent setLocal(
        Operand bytecodeOperand, VirtualRegister machineReg, DataFormat format)
    {
        VariableEvent event;
        WhichType which;
        which.virtualReg = machineReg.offset();
        VariableRepresentation representation;
        representation.operand = bytecodeOperand;
        event.m_kind = SetLocalEvent;
        event.m_dataFormat = format;
        event.m_which = WTFMove(which);
        event.m_representation = WTFMove(representation);
        return event;
    }
    
    static VariableEvent movHint(MinifiedID id, Operand bytecodeReg)
    {
        VariableEvent event;
        WhichType which;
        which.id = id.bits();
        VariableRepresentation representation;
        representation.operand = bytecodeReg;
        event.m_kind = MovHintEvent;
        event.m_which = WTFMove(which);
        event.m_representation = WTFMove(representation);
        return event;
    }
    
    VariableEventKind kind() const
    {
        return static_cast<VariableEventKind>(m_kind);
    }
    
    MinifiedID id() const
    {
        ASSERT(
            m_kind == BirthToFill || m_kind == Fill || m_kind == BirthToSpill || m_kind == Spill
            || m_kind == Death || m_kind == MovHintEvent || m_kind == Birth);
        return MinifiedID::fromBits(m_which.get().id);
    }
    
    DataFormat dataFormat() const
    {
        ASSERT(
            m_kind == BirthToFill || m_kind == Fill || m_kind == BirthToSpill || m_kind == Spill
            || m_kind == SetLocalEvent);
        return m_dataFormat;
    }
    
    MacroAssembler::RegisterID gpr() const
    {
        ASSERT(m_kind == BirthToFill || m_kind == Fill);
        ASSERT(m_dataFormat);
        ASSERT(m_dataFormat != DataFormatDouble);
#if USE(JSVALUE32_64)
        ASSERT(!(m_dataFormat & DataFormatJS));
#endif
        return m_representation.get().gpr;
    }
    
#if USE(JSVALUE32_64)
    MacroAssembler::RegisterID tagGPR() const
    {
        ASSERT(m_kind == BirthToFill || m_kind == Fill);
        ASSERT(m_dataFormat & DataFormatJS);
        return m_representation.get().pair.tagGPR;
    }
    MacroAssembler::RegisterID payloadGPR() const
    {
        ASSERT(m_kind == BirthToFill || m_kind == Fill);
        ASSERT(m_dataFormat & DataFormatJS);
        return m_representation.get().pair.payloadGPR;
    }
#endif // USE(JSVALUE32_64)
    
    MacroAssembler::FPRegisterID fpr() const
    {
        ASSERT(m_kind == BirthToFill || m_kind == Fill);
        ASSERT(m_dataFormat == DataFormatDouble);
        return m_representation.get().fpr;
    }
    
    VirtualRegister spillRegister() const
    {
        ASSERT(m_kind == BirthToSpill || m_kind == Spill);
        return m_representation.get().operand.virtualRegister();
    }

    Operand operand() const
    {
        ASSERT(m_kind == SetLocalEvent || m_kind == MovHintEvent);
        return m_representation.get().operand;
    }
    
    VirtualRegister machineRegister() const
    {
        ASSERT(m_kind == SetLocalEvent);
        return VirtualRegister(m_which.get().virtualReg);
    }
    
    VariableRepresentation variableRepresentation() const { return m_representation.get(); }
    
    void dump(PrintStream&) const;
    
private:
    void dumpFillInfo(const char* name, PrintStream&) const;
    void dumpSpillInfo(const char* name, PrintStream&) const;
    
    union WhichType {
        int virtualReg;
        unsigned id;
    };
    Packed<WhichType> m_which;
    
    // For BirthToFill, Fill:
    //   - The GPR or FPR, or a GPR pair.
    // For BirthToSpill, Spill:
    //   - The virtual register.
    // For MovHintEvent, SetLocalEvent:
    //   - The bytecode operand.
    // For Death:
    //   - Unused.
    Packed<VariableRepresentation> m_representation;
    
    VariableEventKind m_kind;
    DataFormat m_dataFormat { DataFormatNone };
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
