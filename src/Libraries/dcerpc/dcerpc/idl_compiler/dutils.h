/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
**      dutils.h
**
**  FACILITY:
**
**      IDL Compiler Backend
**
**  ABSTRACT:
**
**  Header file for dutils.c
**
**  VERSION: DCE 1.0
*/

#ifndef DUTILS_H
#define DUTILS_H

NAMETABLE_id_t BE_new_local_var_name
(
    char *root
);

char const *BE_get_name
(
    NAMETABLE_id_t id
);

int BE_required_alignment
(
    AST_parameter_n_t *param
);

int BE_resulting_alignment
(
    AST_parameter_n_t *param
);

struct BE_ptr_init_t *BE_new_ptr_init
(
    NAMETABLE_id_t pointer_name,
    AST_type_n_t *pointer_type,
    NAMETABLE_id_t pointee_name,
    AST_type_n_t *pointee_type,
    boolean heap
);

AST_type_n_t *BE_get_type_node
(
    AST_type_k_t kind
);

AST_type_n_t *BE_pointer_type_node
(
    AST_type_n_t *type
);

AST_type_n_t *BE_slice_type_node
(
    AST_type_n_t *type
);

char *BE_first_element_expression
(
    AST_parameter_n_t *param
);

char *BE_count_expression
(
    AST_parameter_n_t *param
);

char *BE_size_expression
(
    AST_parameter_n_t *param
);

void BE_declare_surrogates
(
    AST_operation_n_t *oper,
    AST_parameter_n_t *param
);

void BE_declare_server_surrogates
(
    AST_operation_n_t *oper
);

int BE_num_elts
(
    AST_parameter_n_t *param
);

char *BE_A_expression
(
    AST_parameter_n_t *param,
    int dim
);

char *BE_B_expression
(
    AST_parameter_n_t *param,
    int dim
);

char *BE_Z_expression
(
    AST_parameter_n_t *param,
    int dim
);

AST_parameter_n_t *BE_create_recs
(
    AST_parameter_n_t *params,
    BE_side_t side
);

#ifdef DEBUG_VERBOSE

void traverse(
    AST_parameter_n_t *list,
    int indent
);

void traverse_blocks(
BE_param_blk_t *block
);

#endif

#endif
