/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
**  NAME:
**
**      cstubmts.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  Header file for cstubmts.c
**
*/
void CSPELL_test_status
(
    FILE *fid
);

void CSPELL_test_transceive_status
(
    FILE *fid
);

void DDBE_gen_cstub
(
    FILE *fid,                      /* Handle for emitted C text */
    AST_interface_n_t *p_interface, /* Ptr to AST interface node */
    language_k_t language,          /* Language stub is to interface to */
    char header_name[],         /* Name of header file to be included in stub */
    boolean *cmd_opt,
    void **cmd_val,
    DDBE_vectors_t *dd_vip    /* Data driven BE vector information ptr */
);

void CSPELL_csr_header
(
    FILE *fid,
    char const *p_interface_name,   /* Ptr to name of interface */
    AST_operation_n_t *p_operation, /* Ptr to operation node */
    boolean use_internal_name       /* use internal name if true */
);
