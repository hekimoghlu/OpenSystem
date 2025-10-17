/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 18, 2024.
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
#include "llvm-c/BitWriter.h"
#include "llvm-c/Core.h"
#include "caml/alloc.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"

/*===-- Modules -----------------------------------------------------------===*/

/* Llvm.llmodule -> string -> bool */
CAMLprim value llvm_write_bitcode_file(value M, value Path) {
  int res = LLVMWriteBitcodeToFile((LLVMModuleRef) M, String_val(Path));
  return Val_bool(res == 0);
}

/* ?unbuffered:bool -> Llvm.llmodule -> Unix.file_descr -> bool */
CAMLprim value llvm_write_bitcode_to_fd(value U, value M, value FD) {
  int Unbuffered;
  int res;

  if (U == Val_int(0)) {
    Unbuffered = 0;
  } else {
    Unbuffered = Bool_val(Field(U,0));
  }

  res = LLVMWriteBitcodeToFD((LLVMModuleRef) M, Int_val(FD), 0, Unbuffered);
  return Val_bool(res == 0);
}
