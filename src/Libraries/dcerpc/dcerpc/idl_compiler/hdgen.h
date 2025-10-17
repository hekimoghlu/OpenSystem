/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 26, 2022.
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
**      hdgen.c
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Function prototypes for hdgen.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef HDGEN_H
#define HDGEN_H

void BE_gen_c_header(
    FILE *fid,              /* Handle for emitted C text */
    AST_interface_n_t *ifp, /* Ptr to AST interface node */
    boolean bugs[],         /* List of backward compatibility "bugs" */
    boolean *cmd_opt
);

void CSPELL_type_def
(
    FILE *fid,
    AST_type_n_t *tp,
    boolean spell_tag
);

const char *mapchar
(
    AST_constant_n_t *cp,   /* Constant node with kind == AST_char_const_k */
    boolean warning_flag    /* TRUE => log warning on nonportable escape char */
);
int BE_is_handle_param(AST_parameter_n_t * p);
enum orpc_class_def_type { class_def, proxy_def, stub_def };
void BE_gen_orpc_defs(FILE * fid, AST_interface_n_t * ifp, enum orpc_class_def_type deftype);

#endif
