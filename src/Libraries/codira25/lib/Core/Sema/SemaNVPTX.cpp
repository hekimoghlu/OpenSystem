/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 20, 2022.
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

//===------ SemaNVPTX.cpp ------- NVPTX target-specific routines ----------===//
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
//
//  This file implements semantic analysis functions specific to NVPTX.
//
//===----------------------------------------------------------------------===//

#include "language/Core/Sema/SemaNVPTX.h"
#include "language/Core/Basic/TargetBuiltins.h"
#include "language/Core/Sema/Sema.h"

namespace language::Core {

SemaNVPTX::SemaNVPTX(Sema &S) : SemaBase(S) {}

bool SemaNVPTX::CheckNVPTXBuiltinFunctionCall(const TargetInfo &TI,
                                              unsigned BuiltinID,
                                              CallExpr *TheCall) {
  switch (BuiltinID) {
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_4:
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_8:
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_16:
  case NVPTX::BI__nvvm_cp_async_cg_shared_global_16:
    return SemaRef.checkArgCountAtMost(TheCall, 3);
  }

  return false;
}

} // namespace language::Core
