/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#include "llvm-c/Target.h"
#include "caml/alloc.h"

/* string -> TargetData.t */
CAMLprim LLVMTargetDataRef llvm_targetdata_create(value StringRep) {
  return LLVMCreateTargetData(String_val(StringRep));
}

/* TargetData.t -> [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_targetdata_add(LLVMTargetDataRef TD, LLVMPassManagerRef PM){
  LLVMAddTargetData(TD, PM);
  return Val_unit;
}

/* TargetData.t -> string */
CAMLprim value llvm_targetdata_as_string(LLVMTargetDataRef TD) {
  char *StringRep = LLVMCopyStringRepOfTargetData(TD);
  value Copy = copy_string(StringRep);
  LLVMDisposeMessage(StringRep);
  return Copy;
}

/* TargetData.t -> unit */
CAMLprim value llvm_targetdata_dispose(LLVMTargetDataRef TD) {
  LLVMDisposeTargetData(TD);
  return Val_unit;
}

/* TargetData.t -> Endian.t */
CAMLprim value llvm_byte_order(LLVMTargetDataRef TD) {
  return Val_int(LLVMByteOrder(TD));
}

/* TargetData.t -> int */
CAMLprim value llvm_pointer_size(LLVMTargetDataRef TD) {
  return Val_int(LLVMPointerSize(TD));
}

/* TargetData.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_size_in_bits(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMSizeOfTypeInBits(TD, Ty));
}

/* TargetData.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_store_size(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMStoreSizeOfType(TD, Ty));
}

/* TargetData.t -> Llvm.lltype -> Int64.t */
CAMLprim value llvm_abi_size(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return caml_copy_int64(LLVMABISizeOfType(TD, Ty));
}

/* TargetData.t -> Llvm.lltype -> int */
CAMLprim value llvm_abi_align(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return Val_int(LLVMABIAlignmentOfType(TD, Ty));
}

/* TargetData.t -> Llvm.lltype -> int */
CAMLprim value llvm_stack_align(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return Val_int(LLVMCallFrameAlignmentOfType(TD, Ty));
}

/* TargetData.t -> Llvm.lltype -> int */
CAMLprim value llvm_preferred_align(LLVMTargetDataRef TD, LLVMTypeRef Ty) {
  return Val_int(LLVMPreferredAlignmentOfType(TD, Ty));
}

/* TargetData.t -> Llvm.llvalue -> int */
CAMLprim value llvm_preferred_align_of_global(LLVMTargetDataRef TD,
                                              LLVMValueRef GlobalVar) {
  return Val_int(LLVMPreferredAlignmentOfGlobal(TD, GlobalVar));
}

/* TargetData.t -> Llvm.lltype -> Int64.t -> int */
CAMLprim value llvm_element_at_offset(LLVMTargetDataRef TD, LLVMTypeRef Ty,
                                      value Offset) {
  return Val_int(LLVMElementAtOffset(TD, Ty, Int_val(Offset)));
}

/* TargetData.t -> Llvm.lltype -> int -> Int64.t */
CAMLprim value llvm_offset_of_element(LLVMTargetDataRef TD, LLVMTypeRef Ty,
                                      value Index) {
  return caml_copy_int64(LLVMOffsetOfElement(TD, Ty, Int_val(Index)));
}
