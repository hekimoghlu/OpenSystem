/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_MMA_BASE_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_MMA_BASE_H_

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Policy object describing MmaTensorOp
template <
    /// Warp-level GEMM operator (concept: gemm::warp::Mma)
    typename Operator_,
    /// Padding used for A operand in shared memory (concept: MatrixShape)
    typename SmemPaddingA_,
    /// Padding used for B operand in shared memory (concept: MatrixShape)
    typename SmemPaddingB_,
    /// Number of partitions of K dimension of GEMM
    int PartitionsK = 1>
struct ConvMmaPolicy {
  /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::ConvMmaSimt)
  using Operator = Operator_;

  /// Padding used for A operand in shared memory
  using SmemPaddingA = SmemPaddingA_;

  /// Padding used for B operand in shared memory
  using SmemPaddingB = SmemPaddingB_;

  /// Number of partitions of K dimension
  static int const kPartitionsK = PartitionsK;
};

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: ConvMmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    typename WindowGlobalElement_,
    /// Convolution window storage configuration
    typename ConvWindowConfiguration,
    /// Used for partial specialization
    typename Enable = bool>
class ConvMmaBase {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  //
  // Dependent types
  //

  using WindowGlobalElement = WindowGlobalElement_;
  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount =
      GemmShape<Shape::kM / WarpGemm::kM, Shape::kN / WarpGemm::kN, Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM operations
  static int const kWarpGemmIterations = (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

  /// Tensor reference to the B operand
  using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

  /// Tensor reference to the window lookup in SMEM
  using TensorRefWindow = TensorRef<WindowGlobalElement, layout::PitchLinear>;

  static int const kWindowLength = ConvWindowConfiguration::kTotalAlignedSize;
  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;

    /// Shape of the B matrix operand in shared memory
    using ShapeB = MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                               Shape::kN + Policy::SmemPaddingB::kColumn>;

    using ShapeWindow = layout::PitchLinearShape<kWindowLength, 1>;

   public:
    //
    // Data members
    //

    /// Buffer for A operand
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

    // Buffer for Convolution Window
    AlignedBuffer<WindowGlobalElement, ShapeWindow::kCount> operand_Window;

   public:
    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }

    CUTLASS_HOST_DEVICE
    static typename layout::PitchLinear LayoutWindow(int required_len) {
      return layout::PitchLinear{required_len};
    }

    /// Returns a TensorRef to the convolution window
    CUTLASS_HOST_DEVICE
    TensorRefWindow operand_Window_ref(int required_len) {
      return TensorRefWindow{operand_Window.data(), LayoutWindow(required_len)};
    }
  };

 protected:
  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  ConvMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
        warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_THREADBLOCK_MMA_BASE_H_
