/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#include "llvm-c/Transforms/Scalar.h"
#include "caml/mlvalues.h"
#include "caml/misc.h"

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_constant_propagation(LLVMPassManagerRef PM) {
  LLVMAddConstantPropagationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_sccp(LLVMPassManagerRef PM) {
  LLVMAddSCCPPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_dead_store_elimination(LLVMPassManagerRef PM) {
  LLVMAddDeadStoreEliminationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_aggressive_dce(LLVMPassManagerRef PM) {
  LLVMAddAggressiveDCEPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_scalar_repl_aggregation(LLVMPassManagerRef PM) {
  LLVMAddScalarReplAggregatesPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_scalar_repl_aggregation_ssa(LLVMPassManagerRef PM) {
  LLVMAddScalarReplAggregatesPassSSA(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> int -> unit */
CAMLprim value llvm_add_scalar_repl_aggregation_with_threshold(value threshold,
                                                               LLVMPassManagerRef PM) {
  LLVMAddScalarReplAggregatesPassWithThreshold(PM, Int_val(threshold));
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_ind_var_simplification(LLVMPassManagerRef PM) {
  LLVMAddIndVarSimplifyPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_instruction_combination(LLVMPassManagerRef PM) {
  LLVMAddInstructionCombiningPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_licm(LLVMPassManagerRef PM) {
  LLVMAddLICMPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_loop_unswitch(LLVMPassManagerRef PM) {
  LLVMAddLoopUnswitchPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_loop_unroll(LLVMPassManagerRef PM) {
  LLVMAddLoopUnrollPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_loop_rotation(LLVMPassManagerRef PM) {
  LLVMAddLoopRotatePass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_memory_to_register_promotion(LLVMPassManagerRef PM) {
  LLVMAddPromoteMemoryToRegisterPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_memory_to_register_demotion(LLVMPassManagerRef PM) {
  LLVMAddDemoteMemoryToRegisterPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_reassociation(LLVMPassManagerRef PM) {
  LLVMAddReassociatePass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_jump_threading(LLVMPassManagerRef PM) {
  LLVMAddJumpThreadingPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_cfg_simplification(LLVMPassManagerRef PM) {
  LLVMAddCFGSimplificationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_tail_call_elimination(LLVMPassManagerRef PM) {
  LLVMAddTailCallEliminationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_gvn(LLVMPassManagerRef PM) {
  LLVMAddGVNPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_memcpy_opt(LLVMPassManagerRef PM) {
  LLVMAddMemCpyOptPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_loop_deletion(LLVMPassManagerRef PM) {
  LLVMAddLoopDeletionPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_loop_idiom(LLVMPassManagerRef PM) {
  LLVMAddLoopIdiomPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_lib_call_simplification(LLVMPassManagerRef PM) {
  LLVMAddSimplifyLibCallsPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_verifier(LLVMPassManagerRef PM) {
  LLVMAddVerifierPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_correlated_value_propagation(LLVMPassManagerRef PM) {
  LLVMAddCorrelatedValuePropagationPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_early_cse(LLVMPassManagerRef PM) {
  LLVMAddEarlyCSEPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_lower_expect_intrinsic(LLVMPassManagerRef PM) {
  LLVMAddLowerExpectIntrinsicPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_type_based_alias_analysis(LLVMPassManagerRef PM) {
  LLVMAddTypeBasedAliasAnalysisPass(PM);
  return Val_unit;
}

/* [<Llvm.PassManager.any] Llvm.PassManager.t -> unit */
CAMLprim value llvm_add_basic_alias_analysis(LLVMPassManagerRef PM) {
  LLVMAddBasicAliasAnalysisPass(PM);
  return Val_unit;
}
