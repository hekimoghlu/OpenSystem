/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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
#ifndef LLVM_ANALYSIS_PROFILEDATATYPES_H
#define LLVM_ANALYSIS_PROFILEDATATYPES_H

/* Included by libprofile. */
#if defined(__cplusplus)
extern "C" {
#endif

/* TODO: Strip out unused entries once ProfileInfo etc has been removed. */
enum ProfilingType {
  ArgumentInfo  = 1,   /* The command line argument block */
  FunctionInfo  = 2,   /* Function profiling information  */
  BlockInfo     = 3,   /* Block profiling information     */
  EdgeInfo      = 4,   /* Edge profiling information      */
  PathInfo      = 5,   /* Path profiling information      */
  BBTraceInfo   = 6,   /* Basic block trace information   */
  OptEdgeInfo   = 7    /* Edge profiling information, optimal version */
};

#if defined(__cplusplus)
}
#endif

#endif /* LLVM_ANALYSIS_PROFILEDATATYPES_H */
