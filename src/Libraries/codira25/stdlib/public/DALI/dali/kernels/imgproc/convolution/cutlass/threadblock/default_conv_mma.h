/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_DEFAULT_CONV_MMA_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_DEFAULT_CONV_MMA_H_

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "dali/kernels/imgproc/convolution/cutlass/threadblock/mma_pipelined.h"
#include "dali/kernels/imgproc/convolution/cutlass/threadblock/predicated_tile_iterator.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
#endif  // CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/// Wraps DefaultMma by adding the PositionPredicatedTile iterator and selection
/// between Inner and Outer Conv.
/// Redirects the appropriate iterators to IteratorA (default for IsInnerConv)
///  and IteratorB (default for !IsInnerConv)
template <
    /// Element type for A matrix operand (input in Gmem)
    typename ElementA,
    /// Element type for A matrix operand for computation
    typename ElementCastA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand (input in Gmem)
    typename ElementB,
    /// Element type for B matrix operand for computation
    typename ElementCastB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Convolution window storage configuration
    typename ConvWindowConfiguration,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// If the convolution is computed in the innermost or outer dimension
    bool IsInnerConv = true
    >
struct SpecializedConvMma {
  // Select SMEM iterators that use ElementAccumulator type as storage (and computation)
  // instead of ElementA and ElementB
  using UnderlyingMmaProcessing =
      DefaultMma<ElementCastA, LayoutA, kAlignmentA, ElementCastB, LayoutB, kAlignmentB,
                 ElementAccumulator, LayoutC, OperatorClass, ArchTag, ThreadblockShape, WarpShape,
                 InstructionShape, Stages, Operator, AccumulatorsInRowMajor>;

  // Define the MmaCore components
  using MmaCore = typename UnderlyingMmaProcessing::MmaCore;

  static int const kIsInnerConv = IsInnerConv;

  // PositionPredicatedTileIterators that build matrix on the fly from SMEM
  using IteratorA_outer_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA, LayoutA, 1,
          typename MmaCore::IteratorThreadMapA, ConvWindowConfiguration, kAlignmentA>;

  using IteratorA_regular = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA, LayoutA, 1,
      typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  using IteratorB_inner_conv_smem_ =
      cutlass::transform::threadblock::PositionPredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB, LayoutB, 0,
          typename MmaCore::IteratorThreadMapB, ConvWindowConfiguration, kAlignmentB>;

  using IteratorB_regular = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB, LayoutB, 0,
      typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define iterators over tiles from the A operand
  using IteratorA = std::conditional_t<kIsInnerConv, IteratorA_regular, IteratorA_outer_conv_smem_>;

  // Define iterators over tiles from the B operand
  using IteratorB = std::conditional_t<kIsInnerConv, IteratorB_inner_conv_smem_, IteratorB_regular>;

  // We pass here all the iterators and there is the actual impl of load GMEM->SMEM happening
  // Overwrite the one from UnderlyingMma
  using ThreadblockMma = cutlass::gemm::threadblock::ConvMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA, IteratorB,
      typename MmaCore::SmemIteratorB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, kIsInnerConv, ConvWindowConfiguration>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_DEFAULT_CONV_MMA_H_
