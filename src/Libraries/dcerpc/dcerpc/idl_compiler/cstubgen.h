/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
**      cstubgen.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Function prototypes for cstubgen.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef CSTUBGEN_H
#define CSTUBGEN_H

#include <clihandl.h>

extern BE_handle_info_t BE_handle_info;

void CSPELL_test_status (
    FILE *fid
);

void CSPELL_test_transceive_status
(
    FILE *fid
);

void BE_gen_cstub(
    FILE *fid,                      /* Handle for emitted C text */
    AST_interface_n_t *p_interface, /* Ptr to AST interface node */
    language_k_t language,          /* Language stub is to interface to */
    char header_name[],             /* Name of header file to be included in stub */
    boolean bugs[],                 /* List of backward compatibility "bugs" */
    boolean generate_cepv           /* generate cepv if true */
);

#endif
