/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
/*
**
**  NAME:
**
**      cspell.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Definitions of routines declared in cspell.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef CSPELL_H
#define CSPELL_H

void CSPELL_std_include(
    FILE *fid,
    char header_name[],
    BE_output_k_t filetype,
    int op_count
);

void spell_name(
    FILE *fid,
    NAMETABLE_id_t name
);

void CSPELL_var_decl(
    FILE *fid,
    AST_type_n_t *type,
    NAMETABLE_id_t name
);

void CSPELL_typed_name(
    FILE *fid,
    AST_type_n_t *type,
    NAMETABLE_id_t name,
    AST_type_n_t *in_typedef,
    boolean in_struct,
    boolean spell_tag,
    boolean encoding_services
);

void CSPELL_function_def_header(
    FILE *fid,
    AST_operation_n_t *oper,
    NAMETABLE_id_t name
);

void CSPELL_cast_exp(
    FILE *fid,
    AST_type_n_t *tp
);

void CSPELL_ptr_cast_exp(
    FILE *fid,
    AST_type_n_t *tp
);

void CSPELL_type_exp_simple(
    FILE *fid,
    AST_type_n_t *tp
);

boolean CSPELL_scalar_type_suffix(
    FILE *fid,
    AST_type_n_t *tp
);

void CSPELL_pipe_struct_routine_decl
(
    FILE *fid,
    AST_type_n_t *p_pipe_type,
    BE_pipe_routine_k_t routine_kind,
    boolean cast
);

void CSPELL_midl_compatibility_allocators
(
    FILE *fid
);

void CSPELL_suppress_stub_warnings
(
 FILE *fid
);

void CSPELL_restore_stub_warnings
(
 FILE *fid
);

void CSPELL_restore_stub_warnings
(
 FILE *fid
);

void DDBE_spell_manager_param_cast
(
    FILE *fid,
    AST_type_n_t *tp
);

#endif
