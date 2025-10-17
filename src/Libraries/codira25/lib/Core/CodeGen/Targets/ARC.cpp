/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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

//===- ARC.cpp ------------------------------------------------------------===//
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

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace language::Core;
using namespace language::Core::CodeGen;

// ARC ABI implementation.
namespace {

class ARCABIInfo : public DefaultABIInfo {
  struct CCState {
    unsigned FreeRegs;
  };

public:
  using DefaultABIInfo::DefaultABIInfo;

private:
  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;

  void updateState(const ABIArgInfo &Info, QualType Ty, CCState &State) const {
    if (!State.FreeRegs)
      return;
    if (Info.isIndirect() && Info.getInReg())
      State.FreeRegs--;
    else if (Info.isDirect() && Info.getInReg()) {
      unsigned sz = (getContext().getTypeSize(Ty) + 31) / 32;
      if (sz < State.FreeRegs)
        State.FreeRegs -= sz;
      else
        State.FreeRegs = 0;
    }
  }

  void computeInfo(CGFunctionInfo &FI) const override {
    CCState State;
    // ARC uses 8 registers to pass arguments.
    State.FreeRegs = 8;

    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType());
    updateState(FI.getReturnInfo(), FI.getReturnType(), State);
    for (auto &I : FI.arguments()) {
      I.info = classifyArgumentType(I.type, State.FreeRegs);
      updateState(I.info, I.type, State);
    }
  }

  ABIArgInfo getIndirectByRef(QualType Ty, bool HasFreeRegs) const;
  ABIArgInfo getIndirectByValue(QualType Ty) const;
  ABIArgInfo classifyArgumentType(QualType Ty, uint8_t FreeRegs) const;
  ABIArgInfo classifyReturnType(QualType RetTy) const;
};

class ARCTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  ARCTargetCodeGenInfo(CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<ARCABIInfo>(CGT)) {}
};


ABIArgInfo ARCABIInfo::getIndirectByRef(QualType Ty, bool HasFreeRegs) const {
  return HasFreeRegs ? getNaturalAlignIndirectInReg(Ty)
                     : getNaturalAlignIndirect(
                           Ty, getDataLayout().getAllocaAddrSpace(), false);
}

ABIArgInfo ARCABIInfo::getIndirectByValue(QualType Ty) const {
  // Compute the byval alignment.
  const unsigned MinABIStackAlignInBytes = 4;
  unsigned TypeAlign = getContext().getTypeAlign(Ty) / 8;
  return ABIArgInfo::getIndirect(
      CharUnits::fromQuantity(4),
      /*AddrSpace=*/getDataLayout().getAllocaAddrSpace(),
      /*ByVal=*/true, TypeAlign > MinABIStackAlignInBytes);
}

RValue ARCABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                             QualType Ty, AggValueSlot Slot) const {
  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, /*indirect*/ false,
                          getContext().getTypeInfoInChars(Ty),
                          CharUnits::fromQuantity(4), true, Slot);
}

ABIArgInfo ARCABIInfo::classifyArgumentType(QualType Ty,
                                            uint8_t FreeRegs) const {
  // Handle the generic C++ ABI.
  const RecordType *RT = Ty->getAs<RecordType>();
  if (RT) {
    CGCXXABI::RecordArgABI RAA = getRecordArgABI(RT, getCXXABI());
    if (RAA == CGCXXABI::RAA_Indirect)
      return getIndirectByRef(Ty, FreeRegs > 0);

    if (RAA == CGCXXABI::RAA_DirectInMemory)
      return getIndirectByValue(Ty);
  }

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getOriginalDecl()->getDefinitionOrSelf()->getIntegerType();

  auto SizeInRegs = toolchain::alignTo(getContext().getTypeSize(Ty), 32) / 32;

  if (isAggregateTypeForABI(Ty)) {
    // Structures with flexible arrays are always indirect.
    if (RT &&
        RT->getOriginalDecl()->getDefinitionOrSelf()->hasFlexibleArrayMember())
      return getIndirectByValue(Ty);

    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();

    toolchain::LLVMContext &LLVMContext = getVMContext();

    toolchain::IntegerType *Int32 = toolchain::Type::getInt32Ty(LLVMContext);
    SmallVector<toolchain::Type *, 3> Elements(SizeInRegs, Int32);
    toolchain::Type *Result = toolchain::StructType::get(LLVMContext, Elements);

    return FreeRegs >= SizeInRegs ?
        ABIArgInfo::getDirectInReg(Result) :
        ABIArgInfo::getDirect(Result, 0, nullptr, false);
  }

  if (const auto *EIT = Ty->getAs<BitIntType>())
    if (EIT->getNumBits() > 64)
      return getIndirectByValue(Ty);

  return isPromotableIntegerTypeForABI(Ty)
             ? (FreeRegs >= SizeInRegs ? ABIArgInfo::getExtendInReg(Ty)
                                       : ABIArgInfo::getExtend(Ty))
             : (FreeRegs >= SizeInRegs ? ABIArgInfo::getDirectInReg()
                                       : ABIArgInfo::getDirect());
}

ABIArgInfo ARCABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isAnyComplexType())
    return ABIArgInfo::getDirectInReg();

  // Arguments of size > 4 registers are indirect.
  auto RetSize = toolchain::alignTo(getContext().getTypeSize(RetTy), 32) / 32;
  if (RetSize > 4)
    return getIndirectByRef(RetTy, /*HasFreeRegs*/ true);

  return DefaultABIInfo::classifyReturnType(RetTy);
}

} // End anonymous namespace.

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createARCTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<ARCTargetCodeGenInfo>(CGM.getTypes());
}
