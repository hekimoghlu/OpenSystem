/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

//===----- SemaOpenCL.h --- Semantic Analysis for OpenCL constructs -------===//
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
/// \file
/// This file declares semantic analysis routines for OpenCL.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_CORE_SEMA_SEMAOPENCL_H
#define LANGUAGE_CORE_SEMA_SEMAOPENCL_H

#include "language/Core/AST/ASTFwd.h"
#include "language/Core/Sema/SemaBase.h"

namespace language::Core {
class ParsedAttr;

class SemaOpenCL : public SemaBase {
public:
  SemaOpenCL(Sema &S);

  void handleNoSVMAttr(Decl *D, const ParsedAttr &AL);
  void handleAccessAttr(Decl *D, const ParsedAttr &AL);

  // Handles intel_reqd_sub_group_size.
  void handleSubGroupSize(Decl *D, const ParsedAttr &AL);

  // Performs semantic analysis for the read/write_pipe call.
  // \param S Reference to the semantic analyzer.
  // \param Call A pointer to the builtin call.
  // \return True if a semantic error has been found, false otherwise.
  bool checkBuiltinRWPipe(CallExpr *Call);

  // Performs a semantic analysis on the {work_group_/sub_group_
  //        /_}reserve_{read/write}_pipe
  // \param S Reference to the semantic analyzer.
  // \param Call The call to the builtin function to be analyzed.
  // \return True if a semantic error was found, false otherwise.
  bool checkBuiltinReserveRWPipe(CallExpr *Call);

  bool checkSubgroupExt(CallExpr *Call);

  // Performs a semantic analysis on {work_group_/sub_group_
  //        /_}commit_{read/write}_pipe
  // \param S Reference to the semantic analyzer.
  // \param Call The call to the builtin function to be analyzed.
  // \return True if a semantic error was found, false otherwise.
  bool checkBuiltinCommitRWPipe(CallExpr *Call);

  // Performs a semantic analysis on the call to built-in Pipe
  //        Query Functions.
  // \param S Reference to the semantic analyzer.
  // \param Call The call to the builtin function to be analyzed.
  // \return True if a semantic error was found, false otherwise.
  bool checkBuiltinPipePackets(CallExpr *Call);

  // OpenCL v2.0 s6.13.9 - Address space qualifier functions.
  // Performs semantic analysis for the to_global/local/private call.
  // \param S Reference to the semantic analyzer.
  // \param BuiltinID ID of the builtin function.
  // \param Call A pointer to the builtin call.
  // \return True if a semantic error has been found, false otherwise.
  bool checkBuiltinToAddr(unsigned BuiltinID, CallExpr *Call);

  /// OpenCL C v2.0, s6.13.17 - Enqueue kernel function contains four different
  /// overload formats specified in Table 6.13.17.1.
  /// int enqueue_kernel(queue_t queue,
  ///                    kernel_enqueue_flags_t flags,
  ///                    const ndrange_t ndrange,
  ///                    void (^block)(void))
  /// int enqueue_kernel(queue_t queue,
  ///                    kernel_enqueue_flags_t flags,
  ///                    const ndrange_t ndrange,
  ///                    uint num_events_in_wait_list,
  ///                    clk_event_t *event_wait_list,
  ///                    clk_event_t *event_ret,
  ///                    void (^block)(void))
  /// int enqueue_kernel(queue_t queue,
  ///                    kernel_enqueue_flags_t flags,
  ///                    const ndrange_t ndrange,
  ///                    void (^block)(local void*, ...),
  ///                    uint size0, ...)
  /// int enqueue_kernel(queue_t queue,
  ///                    kernel_enqueue_flags_t flags,
  ///                    const ndrange_t ndrange,
  ///                    uint num_events_in_wait_list,
  ///                    clk_event_t *event_wait_list,
  ///                    clk_event_t *event_ret,
  ///                    void (^block)(local void*, ...),
  ///                    uint size0, ...)
  bool checkBuiltinEnqueueKernel(CallExpr *TheCall);

  /// OpenCL C v2.0, s6.13.17.6 - Check the argument to the
  /// get_kernel_work_group_size
  /// and get_kernel_preferred_work_group_size_multiple builtin functions.
  bool checkBuiltinKernelWorkGroupSize(CallExpr *TheCall);

  bool checkBuiltinNDRangeAndBlock(CallExpr *TheCall);
};

} // namespace language::Core

#endif // LANGUAGE_CORE_SEMA_SEMAOPENCL_H
