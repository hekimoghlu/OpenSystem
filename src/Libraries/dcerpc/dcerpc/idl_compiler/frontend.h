/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
**      frontend.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  Functions exported by frontend.c.
**
**  VERSION: DCE 1.0
**
*/

#include <nidl.h>               /* IDL common defs */
#include <ast.h>                /* Abstract Syntax Tree defs */
#include <nametbl.h>            /* Nametable defs */

#define DCEIDL_DEF   "_DCE_IDL_"

boolean FE_main(
    boolean             *cmd_opt,
    void                **cmd_val,
    STRTAB_str_t        idl_sid,
    AST_interface_n_t   **int_p
);

AST_interface_n_t   *FE_parse_import(
    STRTAB_str_t new_input
);
