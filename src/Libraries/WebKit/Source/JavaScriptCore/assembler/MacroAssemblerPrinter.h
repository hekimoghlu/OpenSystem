/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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

#include "JSCPtrTag.h"
#include "MacroAssembler.h"
#include "Printer.h"
#include "ProbeContext.h"

namespace JSC {

#if ENABLE(ASSEMBLER)

// What is MacroAssembler::print()?
// ===============================
// The MacroAsssembler::print() makes it easy to add print logging
// from JIT compiled code, and can be used to print all types of values
// at runtime e.g. CPU register values being operated on by the compiled
// code.
//
// print() is built on top of MacroAsssembler::probe(), and hence
// inserting logging in JIT compiled code will not perturb register values.
// The only register value that is perturbed is the PC (program counter)
// since there is now more compiled code to do the printing.
//
// How to use the MacroAssembler print()?
// =====================================
// 1. #include "MacroAssemblerPrinter.h" in the JIT file where you want to use print().
//
// 2. Add print() calls like these in your JIT code:
//
//      jit.print("Hello world\n"); // Emits code to print the string.
//
//      CodeBlock* cb = ...;
//      jit.print(cb, "\n");             // Emits code to print the codeBlock value.
//      jit.print(RawPointer(cb), "\n"); // Emits code to print the pointer value.
//
//      RegisterID regID = ...;
//      jit.print(regID, "\n");     // Emits code to print the register value (not the id).
//
//      // Emits code to print all registers. Unlike other items, this prints
//      // multiple lines as follows:
//      //     cpu {
//      //         eax: 0x123456789
//      //         ebx: 0x000000abc
//      //         ...
//      //     }
//      unsigned indentation = 4;
//      jit.print(AllRegisters(indentation));
//
//      jit.print(MemWord<uint8_t>(regID), "\n");   // Emits code to print a byte pointed to by the register.
//      jit.print(MemWord<uint32_t>(regID), "\n");  // Emits code to print a 32-bit word pointed to by the register.
//
//      jit.print(MemWord<uint8_t>(Address(regID, 23), "\n");     // Emits code to print a byte at the address.
//      jit.print(MemWord<intptr_t>(AbsoluteAddress(&cb), "\n");  // Emits code to print an intptr_t sized word at the address.
//
//      jit.print(Memory(reg, 100), "\n");              // Emits code to print a 100 bytes at the address pointed by the register.
//      jit.print(Memory(Address(reg, 4), 100), "\n");  // Emits code to print a 100 bytes at the address.
//
//      // Print multiple things at once. This incurs the probe overhead only once
//      // to print all the items.
//      jit.print("cb:", cb, " regID:", regID, " cpu:\n", AllRegisters());
//
//   The type of values that can be printed is determine by the availability of a
//   specialized Printer template, or a setPrinter() function for the value type.
//
//   Note: print() does not automatically insert a '\n' at the end of the line.
//   If you want a '\n', you'll have to add it explicitly (as in the examples above).

namespace Printer {

struct AllRegisters {
    explicit AllRegisters(unsigned charsToIndent = 0)
        : charsToIndent(charsToIndent)
    { }
    unsigned charsToIndent;
};
struct PCRegister { };

struct Memory {
    using Address = MacroAssembler::Address;
    using AbsoluteAddress = MacroAssembler::AbsoluteAddress;
    using RegisterID = MacroAssembler::RegisterID;

    enum class AddressType {
        Address,
        AbsoluteAddress,
    };

    enum DumpStyle {
        SingleWordDump,
        GenericDump,
    };

    explicit Memory(RegisterID& reg, size_t bytes, DumpStyle style = GenericDump)
        : addressType(AddressType::Address)
        , dumpStyle(style)
        , numBytes(bytes)
    {
        u.address = Address(reg, 0);
    }

    explicit Memory(const Address& address, size_t bytes, DumpStyle style = GenericDump)
        : addressType(AddressType::Address)
        , dumpStyle(style)
        , numBytes(bytes)
    {
        u.address = address;
    }

    explicit Memory(const AbsoluteAddress& address, size_t bytes, DumpStyle style = GenericDump)
        : addressType(AddressType::AbsoluteAddress)
        , dumpStyle(style)
        , numBytes(bytes)
    {
        u.absoluteAddress = address;
    }

    AddressType addressType;
    DumpStyle dumpStyle;
    size_t numBytes;
    union UnionedAddress {
        UnionedAddress() { }

        Address address;
        AbsoluteAddress absoluteAddress;
    } u;
};

template <typename IntType>
struct MemWord : public Memory {
    explicit MemWord(RegisterID& reg)
        : Memory(reg, sizeof(IntType), Memory::SingleWordDump)
    { }

    explicit MemWord(const Address& address)
        : Memory(address, sizeof(IntType), Memory::SingleWordDump)
    { }

    explicit MemWord(const AbsoluteAddress& address)
        : Memory(address, sizeof(IntType), Memory::SingleWordDump)
    { }
};

// Add some specialized printers.

void printAllRegisters(PrintStream&, Context&);
void printPCRegister(PrintStream&, Context&);
void printRegisterID(PrintStream&, Context&);
void printFPRegisterID(PrintStream&, Context&);
void printAddress(PrintStream&, Context&);
void printMemory(PrintStream&, Context&);

template<>
struct Printer<AllRegisters> : public PrintRecord {
    Printer(AllRegisters allRegisters)
        : PrintRecord(static_cast<uintptr_t>(allRegisters.charsToIndent), printAllRegisters)
    { }
};

template<>
struct Printer<PCRegister> : public PrintRecord {
    Printer(PCRegister&)
        : PrintRecord(printPCRegister)
    { }
};

template<>
struct Printer<MacroAssembler::RegisterID> : public PrintRecord {
    Printer(MacroAssembler::RegisterID id)
        : PrintRecord(static_cast<uintptr_t>(id), printRegisterID)
    { }
};

template<>
struct Printer<MacroAssembler::FPRegisterID> : public PrintRecord {
    Printer(MacroAssembler::FPRegisterID id)
        : PrintRecord(static_cast<uintptr_t>(id), printFPRegisterID)
    { }
};

template<>
struct Printer<MacroAssembler::Address> : public PrintRecord {
    Printer(MacroAssembler::Address address)
        : PrintRecord(Data(&address, sizeof(address)), printAddress)
    { }
};

template<>
struct Printer<Memory> : public PrintRecord {
    Printer(Memory memory)
        : PrintRecord(Data(&memory, sizeof(memory)), printMemory)
    { }
};

template<typename IntType>
struct Printer<MemWord<IntType>> : public Printer<Memory> {
    Printer(MemWord<IntType> word)
        : Printer<Memory>(word)
    { }
};

void SYSV_ABI printCallback(Probe::Context&);

} // namespace Printer

template<typename... Arguments>
inline void MacroAssembler::print(Arguments&&... arguments)
{
    auto printRecordList = Printer::makePrintRecordList(std::forward<Arguments>(arguments)...);
    probe(tagCFunction<JITProbePtrTag>(Printer::printCallback), printRecordList);
}

template<typename... Arguments>
inline void MacroAssembler::println(Arguments&&... arguments)
{
    print(std::forward<Arguments>(arguments)..., "\n");
}

inline void MacroAssembler::print(Printer::PrintRecordList* printRecordList)
{
    probe(tagCFunction<JITProbePtrTag>(Printer::printCallback), printRecordList);
}

#endif // ENABLE(ASSEMBLER)

} // namespace JSC
