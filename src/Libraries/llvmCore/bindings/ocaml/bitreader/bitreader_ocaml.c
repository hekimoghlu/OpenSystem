/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
#include "llvm-c/BitReader.h"
#include "caml/alloc.h"
#include "caml/fail.h"
#include "caml/memory.h"


/* Can't use the recommended caml_named_value mechanism for backwards
   compatibility reasons. This is largely equivalent. */
static value llvm_bitreader_error_exn;

CAMLprim value llvm_register_bitreader_exns(value Error) {
  llvm_bitreader_error_exn = Field(Error, 0);
  register_global_root(&llvm_bitreader_error_exn);
  return Val_unit;
}

static void llvm_raise(value Prototype, char *Message) {
  CAMLparam1(Prototype);
  CAMLlocal1(CamlMessage);
  
  CamlMessage = copy_string(Message);
  LLVMDisposeMessage(Message);
  
  raise_with_arg(Prototype, CamlMessage);
  abort(); /* NOTREACHED */
#ifdef CAMLnoreturn
  CAMLnoreturn; /* Silences warnings, but is missing in some versions. */
#endif
}


/*===-- Modules -----------------------------------------------------------===*/

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
CAMLprim value llvm_get_module(LLVMContextRef C, LLVMMemoryBufferRef MemBuf) {
  CAMLparam0();
  CAMLlocal2(Variant, MessageVal);
  char *Message;
  
  LLVMModuleRef M;
  if (LLVMGetBitcodeModuleInContext(C, MemBuf, &M, &Message))
    llvm_raise(llvm_bitreader_error_exn, Message);
  
  CAMLreturn((value) M);
}

/* Llvm.llcontext -> Llvm.llmemorybuffer -> Llvm.llmodule */
CAMLprim value llvm_parse_bitcode(LLVMContextRef C,
                                  LLVMMemoryBufferRef MemBuf) {
  CAMLparam0();
  CAMLlocal2(Variant, MessageVal);
  LLVMModuleRef M;
  char *Message;
  
  if (LLVMParseBitcodeInContext(C, MemBuf, &M, &Message))
    llvm_raise(llvm_bitreader_error_exn, Message);
  
  CAMLreturn((value) M);
}
