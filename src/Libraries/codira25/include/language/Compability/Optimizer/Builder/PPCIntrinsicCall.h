/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 25, 2022.
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

//==-- Builder/PPCIntrinsicCall.h - lowering of PowerPC intrinsics -*-C++-*-==//
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

#ifndef LANGUAGE_COMPABILITY_LOWER_PPCINTRINSICCALL_H
#define LANGUAGE_COMPABILITY_LOWER_PPCINTRINSICCALL_H

#include "language/Compability/Common/static-multimap-view.h"
#include "language/Compability/Optimizer/Builder/IntrinsicCall.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace fir {

/// Enums used to templatize vector intrinsic function generators. Enum does
/// not contain every vector intrinsic, only intrinsics that share generators.
enum class VecOp {
  Abs,
  Add,
  And,
  Anyge,
  Cmpge,
  Cmpgt,
  Cmple,
  Cmplt,
  Convert,
  Ctf,
  Cvf,
  Ld,
  Lde,
  Ldl,
  Lvsl,
  Lvsr,
  Lxv,
  Lxvp,
  Mergeh,
  Mergel,
  Msub,
  Mul,
  Nmadd,
  Perm,
  Permi,
  Sel,
  Sl,
  Sld,
  Sldw,
  Sll,
  Slo,
  Splat,
  Splat_s32,
  Splats,
  Sr,
  Srl,
  Sro,
  St,
  Ste,
  Stxv,
  Stxvp,
  Sub,
  Xl,
  Xlbe,
  Xld2,
  Xlw4,
  Xor,
  Xst,
  Xst_be,
  Xstd2,
  Xstw4
};

/// Enums used to templatize and share lowering of PowerPC MMA intrinsics.
enum class MMAOp {
  AssembleAcc,
  AssemblePair,
  DisassembleAcc,
  DisassemblePair,
  Xxmfacc,
  Xxmtacc,
  Xxsetaccz,
  Pmxvbf16ger2,
  Pmxvbf16ger2nn,
  Pmxvbf16ger2np,
  Pmxvbf16ger2pn,
  Pmxvbf16ger2pp,
  Pmxvf16ger2,
  Pmxvf16ger2nn,
  Pmxvf16ger2np,
  Pmxvf16ger2pn,
  Pmxvf16ger2pp,

  Pmxvf32ger,
  Pmxvf32gernn,
  Pmxvf32gernp,
  Pmxvf32gerpn,
  Pmxvf32gerpp,
  Pmxvf64ger,
  Pmxvf64gernn,
  Pmxvf64gernp,
  Pmxvf64gerpn,
  Pmxvf64gerpp,

  Pmxvi16ger2,
  Pmxvi16ger2pp,
  Pmxvi16ger2s,
  Pmxvi16ger2spp,
  Pmxvi4ger8,
  Pmxvi4ger8pp,
  Pmxvi8ger4,
  Pmxvi8ger4pp,
  Pmxvi8ger4spp,

  Xvbf16ger2,
  Xvbf16ger2nn,
  Xvbf16ger2np,
  Xvbf16ger2pn,
  Xvbf16ger2pp,
  Xvf16ger2,
  Xvf16ger2nn,
  Xvf16ger2np,
  Xvf16ger2pn,
  Xvf16ger2pp,
  Xvf32ger,
  Xvf32gernn,
  Xvf32gernp,
  Xvf32gerpn,
  Xvf32gerpp,
  Xvf64ger,
  Xvf64gernn,
  Xvf64gernp,
  Xvf64gerpn,
  Xvf64gerpp,
  Xvi16ger2,
  Xvi16ger2pp,
  Xvi16ger2s,
  Xvi16ger2spp,
  Xvi4ger8,
  Xvi4ger8pp,
  Xvi8ger4,
  Xvi8ger4pp,
  Xvi8ger4spp,
};

enum class MMAHandlerOp {
  NoOp,
  SubToFunc,
  SubToFuncReverseArgOnLE,
  FirstArgIsResult,
};

// Wrapper struct to encapsulate information for a vector type. Preserves
// sign of eleTy if eleTy is signed/unsigned integer. Helps with vector type
// conversions.
struct VecTypeInfo {
  mlir::Type eleTy;
  uint64_t len;

  mlir::Type toFirVectorType() { return fir::VectorType::get(len, eleTy); }

  // We need a builder to do the signless element conversion.
  mlir::Type toMlirVectorType(mlir::MLIRContext *context) {
    // Will convert to eleTy to signless int if eleTy is signed/unsigned int.
    auto convEleTy{getConvertedElementType(context, eleTy)};
    return mlir::VectorType::get(len, convEleTy);
  }

  bool isFloat32() { return mlir::isa<mlir::Float32Type>(eleTy); }

  bool isFloat64() { return mlir::isa<mlir::Float64Type>(eleTy); }

  bool isFloat() { return isFloat32() || isFloat64(); }
};

//===----------------------------------------------------------------------===//
// Helper functions for argument handling in vector intrinsics.
//===----------------------------------------------------------------------===//

// Returns a VecTypeInfo with element type and length of given fir vector type.
// Preserves signness of fir vector type if element type of integer.
static inline VecTypeInfo getVecTypeFromFirType(mlir::Type firTy) {
  assert(mlir::isa<fir::VectorType>(firTy));
  VecTypeInfo vecTyInfo;
  vecTyInfo.eleTy = mlir::dyn_cast<fir::VectorType>(firTy).getElementType();
  vecTyInfo.len = mlir::dyn_cast<fir::VectorType>(firTy).getLen();
  return vecTyInfo;
}

static inline VecTypeInfo getVecTypeFromFir(mlir::Value firVec) {
  return getVecTypeFromFirType(firVec.getType());
}

// Calculates the vector length and returns a VecTypeInfo with element type and
// length.
static inline VecTypeInfo getVecTypeFromEle(mlir::Value ele) {
  VecTypeInfo vecTyInfo;
  vecTyInfo.eleTy = ele.getType();
  vecTyInfo.len = 16 / (vecTyInfo.eleTy.getIntOrFloatBitWidth() / 8);
  return vecTyInfo;
}

// Converts array of fir vectors to mlir vectors.
static inline toolchain::SmallVector<mlir::Value, 4>
convertVecArgs(fir::FirOpBuilder &builder, mlir::Location loc,
               VecTypeInfo vecTyInfo, toolchain::SmallVector<mlir::Value, 4> args) {
  toolchain::SmallVector<mlir::Value, 4> newArgs;
  auto ty{vecTyInfo.toMlirVectorType(builder.getContext())};
  assert(ty && "unknown mlir vector type");
  for (size_t i = 0; i < args.size(); i++)
    newArgs.push_back(builder.createConvert(loc, ty, args[i]));
  return newArgs;
}

// This overload method is used only if arguments are of different types.
static inline toolchain::SmallVector<mlir::Value, 4>
convertVecArgs(fir::FirOpBuilder &builder, mlir::Location loc,
               toolchain::SmallVectorImpl<VecTypeInfo> &vecTyInfo,
               toolchain::SmallVector<mlir::Value, 4> args) {
  toolchain::SmallVector<mlir::Value, 4> newArgs;
  for (size_t i = 0; i < args.size(); i++) {
    mlir::Type ty{vecTyInfo[i].toMlirVectorType(builder.getContext())};
    assert(ty && "unknown mlir vector type");
    newArgs.push_back(builder.createConvert(loc, ty, args[i]));
  }
  return newArgs;
}

struct PPCIntrinsicLibrary : IntrinsicLibrary {

  // Constructors.
  explicit PPCIntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc)
      : IntrinsicLibrary(builder, loc) {}
  PPCIntrinsicLibrary() = delete;
  PPCIntrinsicLibrary(const PPCIntrinsicLibrary &) = delete;

  // Helper functions for vector element ordering.
  bool isBEVecElemOrderOnLE();
  bool isNativeVecElemOrderOnLE();
  bool changeVecElemOrder();

  // PPC MMA intrinsic generic handler
  template <MMAOp IntrId, MMAHandlerOp HandlerOp>
  void genMmaIntr(toolchain::ArrayRef<fir::ExtendedValue>);

  // PPC intrinsic handlers.
  template <bool isImm>
  void genMtfsf(toolchain::ArrayRef<fir::ExtendedValue>);

  fir::ExtendedValue genVecAbs(mlir::Type resultType,
                               toolchain::ArrayRef<fir::ExtendedValue> args);
  template <VecOp>
  fir::ExtendedValue
  genVecAddAndMulSubXor(mlir::Type resultType,
                        toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecCmp(mlir::Type resultType,
                               toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecConvert(mlir::Type resultType,
                                   toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecAnyCompare(mlir::Type resultType,
                                      toolchain::ArrayRef<fir::ExtendedValue> args);

  fir::ExtendedValue genVecExtract(mlir::Type resultType,
                                   toolchain::ArrayRef<fir::ExtendedValue> args);

  fir::ExtendedValue genVecInsert(mlir::Type resultType,
                                  toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecMerge(mlir::Type resultType,
                                 toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecPerm(mlir::Type resultType,
                                toolchain::ArrayRef<fir::ExtendedValue> args);

  fir::ExtendedValue genVecXlGrp(mlir::Type resultType,
                                 toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecLdCallGrp(mlir::Type resultType,
                                     toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecLdNoCallGrp(mlir::Type resultType,
                                       toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecLvsGrp(mlir::Type resultType,
                                  toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecNmaddMsub(mlir::Type resultType,
                                     toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  fir::ExtendedValue genVecShift(mlir::Type,
                                 toolchain::ArrayRef<fir::ExtendedValue>);

  fir::ExtendedValue genVecSel(mlir::Type resultType,
                               toolchain::ArrayRef<fir::ExtendedValue> args);

  template <VecOp>
  void genVecStore(toolchain::ArrayRef<fir::ExtendedValue>);

  template <VecOp>
  void genVecXStore(toolchain::ArrayRef<fir::ExtendedValue>);

  template <VecOp vop>
  fir::ExtendedValue genVecSplat(mlir::Type resultType,
                                 toolchain::ArrayRef<fir::ExtendedValue> args);

  fir::ExtendedValue genVecXlds(mlir::Type resultType,
                                toolchain::ArrayRef<fir::ExtendedValue> args);
};

const IntrinsicHandler *findPPCIntrinsicHandler(toolchain::StringRef name);

std::pair<const MathOperation *, const MathOperation *>
checkPPCMathOperationsRange(toolchain::StringRef name);

} // namespace fir

#endif // FORTRAN_LOWER_PPCINTRINSICCALL_H
