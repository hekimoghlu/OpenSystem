/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
**  NAME
**      ASTP_DMP.H
**
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**      Header file for the AST Builder Dumper module, ASTP_DMP.C
**
**  VERSION: DCE 1.0
**
*/

#ifndef ASTP_DMP_H
#define ASTP_DMP_H
#ifdef DUMPERS
#include <nidl.h>
#include <ast.h>

/*
 * Exported dump routines
 */

void AST_dump_interface
(
    AST_interface_n_t *if_n_p
);

void AST_dump_operation
(
    AST_operation_n_t *operation_node_ptr,
    int indentation
);

void AST_dump_parameter
(
    AST_parameter_n_t *parameter_node_ptr,
    int indentation
);

void AST_dump_nametable_id
(
    const char   *format_string,
    NAMETABLE_id_t id
);

void AST_dump_parameter
(
    AST_parameter_n_t *param_node_ptr,
    int     indentation
);

void AST_dump_type(
    AST_type_n_t *type_n_p,
    const char *format,
    int indentation
);

void AST_dump_constant
(
    AST_constant_n_t *constant_node_ptr,
    int indentation
);

void AST_enable_hex_dump(void);

#endif /* Dumpers */
#endif /* ASTP_DMP_H */
