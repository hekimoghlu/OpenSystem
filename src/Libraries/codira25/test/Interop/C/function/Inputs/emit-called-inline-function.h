/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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

#ifndef TEST_INTEROP_C_FUNCTION_INPUTS_EMIT_CALLED_INLINE_FUNCTION_H
#define TEST_INTEROP_C_FUNCTION_INPUTS_EMIT_CALLED_INLINE_FUNCTION_H

#ifdef __cplusplus
#define INLINE inline
#else
// When compiling as C, make the functions `static inline`. This is the flavor
// of inline functions for which we require the behavior checked by this test.
// Non-static C inline functions don't require Codira to emit LLVM IR for them
// because the idea is that there will we one `.c` file that declares them
// `extern inline`, causing an out-of-line definition to be emitted to the
// corresponding .o file.
#define INLINE static inline
#endif

INLINE int notCalled() {
  return 42;
}

INLINE int calledTransitively() {
  return 42;
}

#ifdef __cplusplus
class C {
 public:
  int memberFunctionCalledTransitively() {
    return 42;
  }
};

inline int calledTransitivelyFromVarInit() {
  return 42;
}

inline int varUsedFromCodira = calledTransitivelyFromVarInit();
#else
// C only allows constant initializers for variables with static storage
// duration, so there's no way to initialize this with the result of a call to
// an inline method. Just provide _some_ definition of `varImportedToCodira` so
// we can import it in the test.
static int varUsedFromCodira = 42;
#endif

INLINE int calledFromCodira() {
#ifdef __cplusplus
  C c;
  return calledTransitively() + c.memberFunctionCalledTransitively();
#else
  return calledTransitively();
#endif
}

#endif
