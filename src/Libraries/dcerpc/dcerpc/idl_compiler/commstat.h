/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
**      commstat.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Data types and function prototypes for commstat.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef COMMSTAT_H
#define COMMSTAT_H

typedef enum {BE_stat_param_k, BE_stat_result_k, BE_stat_addl_k,
              BE_stat_except_k} BE_stat_k_t;

typedef struct BE_stat_info_t {
    BE_stat_k_t type;
    NAMETABLE_id_t name;
} BE_stat_info_t;

void BE_get_comm_stat_info
(
    AST_operation_n_t *p_operation,
    BE_stat_info_t *p_comm_stat_info
);

void BE_get_fault_stat_info
(
    AST_operation_n_t *p_operation,
    BE_stat_info_t *p_fault_stat_info
);

void CSPELL_receive_fault
(
    FILE *fid
);

void CSPELL_return_status
(
    FILE *fid,
    BE_stat_info_t *p_comm_stat_info,
    BE_stat_info_t *p_fault_stat_info,
    const char *status_var_name,
    const char *result_param_name,
    int num_user_exceptions,
    const char *IDL_msp_name
);

#endif
