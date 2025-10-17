/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 1, 2025.
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
#include "llvm-c/Analysis.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"


/* Llvm.llmodule -> string option */
CAMLprim value llvm_verify_module(LLVMModuleRef M) {
  CAMLparam0();
  CAMLlocal2(String, Option);
  
  char *Message;
  int Result = LLVMVerifyModule(M, LLVMReturnStatusAction, &Message);
  
  if (0 == Result) {
    Option = Val_int(0);
  } else {
    Option = alloc(1, 0);
    String = copy_string(Message);
    Store_field(Option, 0, String);
  }
  
  LLVMDisposeMessage(Message);
  
  CAMLreturn(Option);
}

/* Llvm.llvalue -> bool */
CAMLprim value llvm_verify_function(LLVMValueRef Fn) {
  return Val_bool(LLVMVerifyFunction(Fn, LLVMReturnStatusAction) == 0);
}

/* Llvm.llmodule -> unit */
CAMLprim value llvm_assert_valid_module(LLVMModuleRef M) {
  LLVMVerifyModule(M, LLVMAbortProcessAction, 0);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
CAMLprim value llvm_assert_valid_function(LLVMValueRef Fn) {
  LLVMVerifyFunction(Fn, LLVMAbortProcessAction);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
CAMLprim value llvm_view_function_cfg(LLVMValueRef Fn) {
  LLVMViewFunctionCFG(Fn);
  return Val_unit;
}

/* Llvm.llvalue -> unit */
CAMLprim value llvm_view_function_cfg_only(LLVMValueRef Fn) {
  LLVMViewFunctionCFGOnly(Fn);
  return Val_unit;
}
