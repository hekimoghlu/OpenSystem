/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_DEFAULT_CONV_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_DEFAULT_CONV_H_

#include "cutlass/cutlass.h"

#include "cutlass/arch/wmma.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "dali/kernels/imgproc/convolution/cutlass/kernel/gemm.h"
#include "dali/kernels/imgproc/convolution/cutlass/threadblock/default_conv_mma.h"
#include "dali/kernels/imgproc/convolution/cutlass/threadblock/predicated_tile_iterator.h"



#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif  // CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

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
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// If the convolution is computed in the innermost or outer dimension
    bool IsInnerConv = true>
struct DefaultConv {
  using UnderlyingConv =
      DefaultGemm<ElementCastA, LayoutA, kAlignmentA, ElementCastB, LayoutB, kAlignmentB, ElementC,
                  layout::RowMajor, ElementAccumulator, OperatorClass, ArchTag, ThreadblockShape,
                  WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle, Stages,
                  SplitKSerial, Operator>;

  using Mma = typename cutlass::gemm::threadblock::SpecializedConvMma<
      ElementA, ElementCastA, LayoutA, kAlignmentA, ElementB, ElementCastB, LayoutB, kAlignmentB,
      ConvWindowConfiguration, ElementAccumulator, LayoutC, OperatorClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape, Stages, Operator, false,
      IsInnerConv>::ThreadblockMma;

  /// Define the epilogue
  using Epilogue = typename UnderlyingConv::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::Conv<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_KERNEL_DEFAULT_CONV_H_
