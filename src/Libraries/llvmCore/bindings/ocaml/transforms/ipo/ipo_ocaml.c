/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 13, 2025.
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
#include "llvm-c/Transforms/IPO.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_argument_promotion(LLVMPassManagerRef PM) {
  LLVMAddArgumentPromotionPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_constant_merge(LLVMPassManagerRef PM) {
  LLVMAddConstantMergePass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_dead_arg_elimination(LLVMPassManagerRef PM) {
  LLVMAddDeadArgEliminationPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_function_attrs(LLVMPassManagerRef PM) {
  LLVMAddFunctionAttrsPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_function_inlining(LLVMPassManagerRef PM) {
  LLVMAddFunctionInliningPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_always_inliner_pass(LLVMPassManagerRef PM) {
  LLVMAddAlwaysInlinerPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_global_dce(LLVMPassManagerRef PM) {
  LLVMAddGlobalDCEPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_global_optimizer(LLVMPassManagerRef PM) {
  LLVMAddGlobalOptimizerPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_ipc_propagation(LLVMPassManagerRef PM) {
  LLVMAddIPConstantPropagationPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_prune_eh(LLVMPassManagerRef PM) {
  LLVMAddPruneEHPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_ipsccp(LLVMPassManagerRef PM) {
  LLVMAddIPSCCPPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> bool -> unit */
CAMLprim value llvm_add_internalize(LLVMPassManagerRef PM, value AllButMain) {
  LLVMAddInternalizePass(PM, Bool_val(AllButMain));
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_strip_dead_prototypes(LLVMPassManagerRef PM) {
  LLVMAddStripDeadPrototypesPass(PM);
  return Val_unit;
}

/* [`Module] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_strip_symbols(LLVMPassManagerRef PM) {
  LLVMAddStripSymbolsPass(PM);
  return Val_unit;
}
