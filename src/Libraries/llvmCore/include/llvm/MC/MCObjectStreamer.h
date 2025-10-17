/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

//===- MCObjectStreamer.h - MCStreamer Object File Interface ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCOBJECTSTREAMER_H
#define LLVM_MC_MCOBJECTSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {
class MCAssembler;
class MCCodeEmitter;
class MCSectionData;
class MCExpr;
class MCFragment;
class MCDataFragment;
class MCAsmBackend;
class raw_ostream;

/// \brief Streaming object file generation interface.
///
/// This class provides an implementation of the MCStreamer interface which is
/// suitable for use with the assembler backend. Specific object file formats
/// are expected to subclass this interface to implement directives specific
/// to that file format or custom semantics expected by the object writer
/// implementation.
class MCObjectStreamer : public MCStreamer {
  MCAssembler *Assembler;
  MCSectionData *CurSectionData;

  virtual void EmitInstToData(const MCInst &Inst) = 0;
  virtual void EmitCFIStartProcImpl(MCDwarfFrameInfo &Frame);
  virtual void EmitCFIEndProcImpl(MCDwarfFrameInfo &Frame);

protected:
  MCObjectStreamer(MCContext &Context, MCAsmBackend &TAB,
                   raw_ostream &_OS, MCCodeEmitter *_Emitter);
  MCObjectStreamer(MCContext &Context, MCAsmBackend &TAB,
                   raw_ostream &_OS, MCCodeEmitter *_Emitter,
                   MCAssembler *_Assembler);
  ~MCObjectStreamer();

  MCSectionData *getCurrentSectionData() const {
    return CurSectionData;
  }

  MCFragment *getCurrentFragment() const;

  /// Get a data fragment to write into, creating a new one if the current
  /// fragment is not a data fragment.
  MCDataFragment *getOrCreateDataFragment() const;

  const MCExpr *AddValueSymbols(const MCExpr *Value);

public:
  MCAssembler &getAssembler() { return *Assembler; }

  /// @name MCStreamer Interface
  /// @{

  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                             unsigned AddrSpace);
  virtual void EmitULEB128Value(const MCExpr *Value);
  virtual void EmitSLEB128Value(const MCExpr *Value);
  virtual void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol);
  virtual void ChangeSection(const MCSection *Section);
  virtual void EmitInstruction(const MCInst &Inst);
  virtual void EmitInstToFragment(const MCInst &Inst);
  virtual bool EmitValueToOffset(const MCExpr *Offset, unsigned char Value);
  virtual void EmitDwarfAdvanceLineAddr(int64_t LineDelta,
                                        const MCSymbol *LastLabel,
                                        const MCSymbol *Label,
                                        unsigned PointerSize);
  virtual void EmitDwarfAdvanceFrameAddr(const MCSymbol *LastLabel,
                                         const MCSymbol *Label);
  virtual void EmitGPRel32Value(const MCExpr *Value);
  virtual void EmitGPRel64Value(const MCExpr *Value);
  virtual void EmitFill(uint64_t NumBytes, uint8_t FillValue,
                        unsigned AddrSpace);
  virtual void FinishImpl();

  /// @}
};

} // end namespace llvm

#endif
