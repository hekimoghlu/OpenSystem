/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
**      propagat.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  Declaration of functions exported by propagat.c.
**
**  VERSION: DCE 1.0
**
*/

#ifndef PROPAGATH_INCL
#define PROPAGATH_INCL

#include <ast.h>                /* Abstract Syntax Tree defs */

/*
**  P R O P _ m a i n
**
**  Main routine for propagation component of IDL compiler.
**  Propagates attributes and other information throughout the AST.
**  Much of propagation is done by earlier compiler phases - this
**  routine handles any propagation that is easier to do if saved
**  until after the parsing/AST-building phases are complete.
*/

extern boolean PROP_main(       /* Returns true on success */
    boolean     *cmd_opt_arr,   /* [in] Array of command option flags */
    void        **cmd_val_arr,  /* [in] Array of command option values */
    AST_interface_n_t *int_p    /* [in] Ptr to AST interface node */
);

#ifndef mips
void PROP_set_type_attr
(
    AST_type_n_t *type_node_ptr,
    AST_flags_t  type_attr
);
#endif

#endif
