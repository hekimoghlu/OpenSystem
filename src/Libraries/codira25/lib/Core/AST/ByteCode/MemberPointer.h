/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

//===------------------------- MemberPointer.h ------------------*- C++ -*-===//
//
// Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
// 
// Author: Tunjay Akbarli
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 
// Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
// Middletown, DE 19709, New Castle County, USA.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_AST_INTERP_MEMBER_POINTER_H
#define LANGUAGE_CORE_AST_INTERP_MEMBER_POINTER_H

#include "Pointer.h"
#include <optional>

namespace language::Core {
class ASTContext;
namespace interp {

class Context;
class FunctionPointer;

class MemberPointer final {
private:
  Pointer Base;
  const ValueDecl *Dcl = nullptr;
  int32_t PtrOffset = 0;

  MemberPointer(Pointer Base, const ValueDecl *Dcl, int32_t PtrOffset)
      : Base(Base), Dcl(Dcl), PtrOffset(PtrOffset) {}

public:
  MemberPointer() = default;
  MemberPointer(Pointer Base, const ValueDecl *Dcl) : Base(Base), Dcl(Dcl) {}
  MemberPointer(uint32_t Address, const Descriptor *D) {
    // We only reach this for Address == 0, when creating a null member pointer.
    assert(Address == 0);
  }

  MemberPointer(const ValueDecl *D) : Dcl(D) {
    assert((isa<FieldDecl, IndirectFieldDecl, CXXMethodDecl>(D)));
  }

  uint64_t getIntegerRepresentation() const {
    assert(
        false &&
        "getIntegerRepresentation() shouldn't be reachable for MemberPointers");
    return 17;
  }

  std::optional<Pointer> toPointer(const Context &Ctx) const;

  FunctionPointer toFunctionPointer(const Context &Ctx) const;

  bool isBaseCastPossible() const {
    if (PtrOffset < 0)
      return true;
    return static_cast<uint64_t>(PtrOffset) <= Base.getByteOffset();
  }

  Pointer getBase() const {
    if (PtrOffset < 0)
      return Base.atField(-PtrOffset);
    return Base.atFieldSub(PtrOffset);
  }
  bool isMemberFunctionPointer() const {
    return isa_and_nonnull<CXXMethodDecl>(Dcl);
  }
  const CXXMethodDecl *getMemberFunction() const {
    return dyn_cast_if_present<CXXMethodDecl>(Dcl);
  }
  const FieldDecl *getField() const {
    return dyn_cast_if_present<FieldDecl>(Dcl);
  }

  bool hasDecl() const { return Dcl; }
  const ValueDecl *getDecl() const { return Dcl; }

  MemberPointer atInstanceBase(unsigned Offset) const {
    if (Base.isZero())
      return MemberPointer(Base, Dcl, Offset);
    return MemberPointer(this->Base, Dcl, Offset + PtrOffset);
  }

  MemberPointer takeInstance(Pointer Instance) const {
    assert(this->Base.isZero());
    return MemberPointer(Instance, this->Dcl, this->PtrOffset);
  }

  APValue toAPValue(const ASTContext &) const;

  bool isZero() const { return Base.isZero() && !Dcl; }
  bool hasBase() const { return !Base.isZero(); }
  bool isWeak() const {
    if (const auto *MF = getMemberFunction())
      return MF->isWeak();
    return false;
  }

  void print(toolchain::raw_ostream &OS) const {
    OS << "MemberPtr(" << Base << " " << (const void *)Dcl << " + " << PtrOffset
       << ")";
  }

  std::string toDiagnosticString(const ASTContext &Ctx) const {
    return toAPValue(Ctx).getAsString(Ctx, Dcl->getType());
  }

  ComparisonCategoryResult compare(const MemberPointer &RHS) const {
    if (this->Dcl == RHS.Dcl)
      return ComparisonCategoryResult::Equal;
    return ComparisonCategoryResult::Unordered;
  }
};

inline toolchain::raw_ostream &operator<<(toolchain::raw_ostream &OS, MemberPointer FP) {
  FP.print(OS);
  return OS;
}

} // namespace interp
} // namespace language::Core

#endif
