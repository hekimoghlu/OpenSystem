/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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
**
**      ASTP_BLD_GLOBALS.C
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Defines global variables used by the parser and abstract
**      sytax tree (AST) builder modules.
**
**  VERSION: DCE 1.0
**
*/

#include <nidl.h>
#include <astp.h>

/*
 * External variables defined here, exported in ASTP.H
 * Theses externals are shared between the AST builder modules
 */

/*
 *  Interface Attributes
 */

/*
 *  Operation, Parameter, Type Attributes
 */

AST_type_n_t        *ASTP_transmit_as_type = NULL;
AST_type_n_t        *ASTP_switch_type = NULL;
AST_case_label_n_t  *ASTP_case = NULL;

/*
 *  Interface just parsed
 */
AST_interface_n_t *the_interface = NULL;

//centeris wfu
AST_cpp_quote_n_t *global_cppquotes = NULL;

AST_cpp_quote_n_t *global_cppquotes_post = NULL;

AST_import_n_t *global_imports = NULL;

/*
 * List head for saved context for field
 * attributes forward referenced parameters.
 */
ASTP_field_ref_ctx_t *ASTP_field_ref_ctx_list = NULL;

/*
 * List head for referenced struct/union tags.
 */
ASTP_tag_ref_n_t *ASTP_tag_ref_list = NULL;

/*
 *  Control for parser
 */
boolean ASTP_parsing_main_idl = TRUE;

/*
 *  Builtin in constants
 */

AST_constant_n_t    *zero_constant_p = NULL;

/*
 * Builtin base types
 */
AST_type_n_t    *ASTP_char_ptr = NULL,
                *ASTP_boolean_ptr = NULL,
                *ASTP_byte_ptr = NULL,
                *ASTP_void_ptr = NULL,
                *ASTP_handle_ptr = NULL,
                *ASTP_short_float_ptr = NULL,
                *ASTP_long_float_ptr = NULL,
                *ASTP_small_int_ptr = NULL,
                *ASTP_short_int_ptr = NULL,
                *ASTP_long_int_ptr = NULL,
                *ASTP_hyper_int_ptr = NULL,
                *ASTP_small_unsigned_ptr = NULL,
                *ASTP_short_unsigned_ptr = NULL,
                *ASTP_long_unsigned_ptr = NULL,
                *ASTP_hyper_unsigned_ptr = NULL;

/* Default tag for union */
NAMETABLE_id_t  ASTP_tagged_union_id;
