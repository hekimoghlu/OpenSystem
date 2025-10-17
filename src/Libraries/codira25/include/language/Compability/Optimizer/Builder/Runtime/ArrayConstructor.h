/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 21, 2025.
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

//===- ArrayConstructor.h - array constructor runtime API calls -*- C++ -*-===//
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

#ifndef LANGUAGE_COMPABILITY_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H
#define LANGUAGE_COMPABILITY_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H

namespace mlir {
class Value;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

mlir::Value genInitArrayConstructorVector(mlir::Location loc,
                                          fir::FirOpBuilder &builder,
                                          mlir::Value toBox,
                                          mlir::Value useValueLengthParameters);

void genPushArrayConstructorValue(mlir::Location loc,
                                  fir::FirOpBuilder &builder,
                                  mlir::Value arrayConstructorVector,
                                  mlir::Value fromBox);

void genPushArrayConstructorSimpleScalar(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value arrayConstructorVector,
                                         mlir::Value fromAddress);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_ARRAYCONSTRUCTOR_H
