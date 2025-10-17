/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
    \brief Definitions for GEMM structures
*/


#ifndef DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_DEFAULT_CONV_CONFIGURATION_H_
#define DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_DEFAULT_CONV_CONFIGURATION_H_

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/gemm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/// Propagate the default configurations
template <typename OperatorClass, typename ArchTag, typename ElementA, typename ElementB,
          typename ElementC, typename ElementAccumulator>
struct DefaultConvConfiguration {
  using UnderlyingConv = DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA, ElementB,
                                                  ElementC, ElementAccumulator>;

  static int const kAlignmentA = UnderlyingConv::kAlignmentA;
  static int const kAlignmentB = UnderlyingConv::kAlignmentB;

  using ThreadblockShape = typename UnderlyingConv::ThreadblockShape;
  using WarpShape = typename UnderlyingConv::WarpShape;
  using InstructionShape = typename UnderlyingConv::InstructionShape;
  static int const kStages = UnderlyingConv::kStages;

  using EpilogueOutputOp = typename UnderlyingConv::EpilogueOutputOp;

  using Operator = typename UnderlyingConv::Operator;
};

////////////////////////////////////////////////////////////////////////////////
}  // namespace device
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////


#endif  // DALI_KERNELS_IMGPROC_CONVOLUTION_CUTLASS_DEVICE_DEFAULT_CONV_CONFIGURATION_H_
