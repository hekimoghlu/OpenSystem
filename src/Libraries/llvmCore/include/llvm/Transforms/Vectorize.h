/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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

//===-- Vectorize.h - Vectorization Transformations -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Vectorize transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_H
#define LLVM_TRANSFORMS_VECTORIZE_H

namespace llvm {
class BasicBlock;
class BasicBlockPass;

//===----------------------------------------------------------------------===//
/// @brief Vectorize configuration.
struct VectorizeConfig {
  //===--------------------------------------------------------------------===//
  // Target architecture related parameters

  /// @brief The size of the native vector registers.
  unsigned VectorBits;

  /// @brief Vectorize boolean values.
  bool VectorizeBools;

  /// @brief Vectorize integer values.
  bool VectorizeInts;

  /// @brief Vectorize floating-point values.
  bool VectorizeFloats;

  /// @brief Vectorize pointer values.
  bool VectorizePointers;

  /// @brief Vectorize casting (conversion) operations.
  bool VectorizeCasts;

  /// @brief Vectorize floating-point math intrinsics.
  bool VectorizeMath;

  /// @brief Vectorize the fused-multiply-add intrinsic.
  bool VectorizeFMA;

  /// @brief Vectorize select instructions.
  bool VectorizeSelect;

  /// @brief Vectorize comparison instructions.
  bool VectorizeCmp;

  /// @brief Vectorize getelementptr instructions.
  bool VectorizeGEP;

  /// @brief Vectorize loads and stores.
  bool VectorizeMemOps;

  /// @brief Only generate aligned loads and stores.
  bool AlignedOnly;

  //===--------------------------------------------------------------------===//
  // Misc parameters

  /// @brief The required chain depth for vectorization.
  unsigned ReqChainDepth;

  /// @brief The maximum search distance for instruction pairs.
  unsigned SearchLimit;

  /// @brief The maximum number of candidate pairs with which to use a full
  ///        cycle check.
  unsigned MaxCandPairsForCycleCheck;

  /// @brief Replicating one element to a pair breaks the chain.
  bool SplatBreaksChain;

  /// @brief The maximum number of pairable instructions per group.
  unsigned MaxInsts;

  /// @brief The maximum number of pairing iterations.
  unsigned MaxIter;

  /// @brief Don't try to form odd-length vectors.
  bool Pow2LenOnly;

  /// @brief Don't boost the chain-depth contribution of loads and stores.
  bool NoMemOpBoost;

  /// @brief Use a fast instruction dependency analysis.
  bool FastDep;

  /// @brief Initialize the VectorizeConfig from command line options.
  VectorizeConfig();
};

//===----------------------------------------------------------------------===//
//
// BBVectorize - A basic-block vectorization pass.
//
BasicBlockPass *
createBBVectorizePass(const VectorizeConfig &C = VectorizeConfig());

//===----------------------------------------------------------------------===//
/// @brief Vectorize the BasicBlock.
///
/// @param BB The BasicBlock to be vectorized
/// @param P  The current running pass, should require AliasAnalysis and
///           ScalarEvolution. After the vectorization, AliasAnalysis,
///           ScalarEvolution and CFG are preserved.
///
/// @return True if the BB is changed, false otherwise.
///
bool vectorizeBasicBlock(Pass *P, BasicBlock &BB,
                         const VectorizeConfig &C = VectorizeConfig());

} // End llvm namespace

#endif
