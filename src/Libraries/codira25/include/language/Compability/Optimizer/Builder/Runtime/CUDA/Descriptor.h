/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

//===-- Descriptor.h - CUDA descritpor runtime API calls --------*- C++ -*-===//
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

#ifndef LANGUAGE_COMPABILITY_OPTIMIZER_BUILDER_RUNTIME_CUDA_DESCRIPTOR_H_
#define LANGUAGE_COMPABILITY_OPTIMIZER_BUILDER_RUNTIME_CUDA_DESCRIPTOR_H_

#include "mlir/IR/Value.h"

namespace mlir {
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime::cuda {

/// Generate runtime call to sync the doublce descriptor referenced by
/// \p hostPtr.
void genSyncGlobalDescriptor(fir::FirOpBuilder &builder, mlir::Location loc,
                             mlir::Value hostPtr);

/// Generate runtime call to check the section of a descriptor and raise an
/// error if it is not contiguous.
void genDescriptorCheckSection(fir::FirOpBuilder &builder, mlir::Location loc,
                               mlir::Value desc);

/// Generate runtime call to set the allocator index in the descriptor.
void genSetAllocatorIndex(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value desc, mlir::Value index);

} // namespace fir::runtime::cuda

#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_CUDA_DESCRIPTOR_H_
