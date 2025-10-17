/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

//===-- ARMMCExpr.h - ARM specific MC expression classes --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ARMMCEXPR_H
#define ARMMCEXPR_H

#include "llvm/MC/MCExpr.h"

namespace llvm {

class ARMMCExpr : public MCTargetExpr {
public:
  enum VariantKind {
    VK_ARM_None,
    VK_ARM_HI16,  // The R_ARM_MOVT_ABS relocation (:upper16: in the .s file)
    VK_ARM_LO16   // The R_ARM_MOVW_ABS_NC relocation (:lower16: in the .s file)
  };

private:
  const VariantKind Kind;
  const MCExpr *Expr;

  explicit ARMMCExpr(VariantKind _Kind, const MCExpr *_Expr)
    : Kind(_Kind), Expr(_Expr) {}

public:
  /// @name Construction
  /// @{

  static const ARMMCExpr *Create(VariantKind Kind, const MCExpr *Expr,
                                      MCContext &Ctx);

  static const ARMMCExpr *CreateUpper16(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_ARM_HI16, Expr, Ctx);
  }

  static const ARMMCExpr *CreateLower16(const MCExpr *Expr, MCContext &Ctx) {
    return Create(VK_ARM_LO16, Expr, Ctx);
  }

  /// @}
  /// @name Accessors
  /// @{

  /// getOpcode - Get the kind of this expression.
  VariantKind getKind() const { return Kind; }

  /// getSubExpr - Get the child of this expression.
  const MCExpr *getSubExpr() const { return Expr; }

  /// @}

  void PrintImpl(raw_ostream &OS) const;
  bool EvaluateAsRelocatableImpl(MCValue &Res,
                                 const MCAsmLayout *Layout) const;
  void AddValueSymbols(MCAssembler *) const;
  const MCSection *FindAssociatedSection() const {
    return getSubExpr()->FindAssociatedSection();
  }

  static bool classof(const MCExpr *E) {
    return E->getKind() == MCExpr::Target;
  }

  static bool classof(const ARMMCExpr *) { return true; }

};
} // end namespace llvm

#endif
