/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
**      genpipes.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Function prototypes for genpipes.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef GENPIPES_H
#define GENPIPES_H

#define BE_FINISHED_WITH_PIPES -32767

void BE_spell_pipe_struct_name
(
    AST_parameter_n_t *p_parameter,
    char pipe_struct_name[]
);

void CSPELL_init_server_pipes
(
    FILE *fid,
    AST_operation_n_t *p_operation,
    long *p_first_pipe      /* ptr to index and direction of first pipe */
);

void CSPELL_pipe_support_header
(
    FILE *fid,
    AST_type_n_t *p_pipe_type,
    BE_pipe_routine_k_t push_or_pull,
    boolean in_header
);

void BE_gen_pipe_routines
(
    FILE *fid,
    AST_interface_n_t *p_interface
);

void BE_gen_pipe_routine_decls
(
    FILE *fid,
    AST_interface_n_t *p_interface
);

void CSPELL_pipe_base_cast_exp
(
    FILE *fid,
    AST_type_n_t *p_type
);

void CSPELL_pipe_base_type_exp
(
    FILE *fid,
    AST_type_n_t *p_type
);

void BE_undec_piped_arrays
(
    AST_parameter_n_t *flat
);

#endif
