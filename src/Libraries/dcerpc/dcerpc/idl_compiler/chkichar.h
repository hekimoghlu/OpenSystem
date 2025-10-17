/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
**      chkichar.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  IDL Compiler Semantic Checking for international character support.
**
*/

extern void CHK_param_cs(
    AST_parameter_n_t   *param_p,       /* [in] Ptr to AST parameter node */
    AST_type_n_t        *type_p         /* [in] Parameter type */
);

extern void CHK_op_cs(
    AST_operation_n_t   *op_p           /* [in] Ptr to AST operation node */
);

extern void CHK_field_cs(
    AST_field_n_t       *field_p        /* [in] Ptr to AST field node */
);

extern void CHK_pipe_base_type_cs(
    AST_pipe_n_t        *pipe_p,        /* [in] Ptr to AST pipe node */
    AST_interface_n_t   *int_p          /* [in] Ptr to interface node */
);

extern void CHK_type_cs(
    AST_type_n_t        *top_type_p,    /* [in] Top-level presented type */
    AST_type_n_t        *type_p,        /* [in] Ptr to AST type node */
    AST_interface_n_t   *int_p          /* [in] Ptr to interface node */
);
