/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
**      NAMETBL.H
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**      Header file for Name Table module, NAMETBL.C
**
**  VERSION: DCE 1.0
**
*/

#ifndef  nametable_incl
#define nametable_incl

/*
** IDL.H needs the definition of STRTAB_str_t, so put it first.
*/

/*
 * it is opaque enough but gives the compiler
 * the oportunity to check the type when neeed
 */
typedef struct NAMETABLE_n_t * NAMETABLE_id_t;
#define NAMETABLE_NIL_ID NULL

typedef NAMETABLE_id_t  STRTAB_str_t ;
#define STRTAB_NULL_STR  NULL

#include <nidl.h>

#define NAMETABLE_id_too_long         1
#define NAMETABLE_no_space            2
#define NAMETABLE_different_casing    3
#define NAMETABLE_string_to_long      4
#define NAMETABLE_bad_id_len          5
#define NAMETABLE_bad_string_len      6

/*
 * This constant needs to be arbitrarily large since derived names added to
 * the nametable can get arbitrarily large, e.g. with nested structures.
 */
#define max_string_len                4096

NAMETABLE_id_t NAMETABLE_add_id(
    const char *id
);

NAMETABLE_id_t NAMETABLE_lookup_id(
    const char *id
);

void NAMETABLE_id_to_string(
    NAMETABLE_id_t NAMETABLE_id,
    const char **str_ptr
);

boolean NAMETABLE_add_binding(
    NAMETABLE_id_t id,
    const void * binding
);

const void* NAMETABLE_lookup_binding(
    NAMETABLE_id_t identifier
);

boolean NAMETABLE_add_tag_binding(
    NAMETABLE_id_t id,
    const void * binding
);

const void* NAMETABLE_lookup_tag_binding(
    NAMETABLE_id_t identifier
);

const void* NAMETABLE_lookup_local(
    NAMETABLE_id_t identifier
);

void  NAMETABLE_push_level(
    void
);

void  NAMETABLE_pop_level(
    void
);

void  NAMETABLE_set_temp_name_mode (
    void
);

void  NAMETABLE_set_perm_name_mode (
    void
);

void  NAMETABLE_clear_temp_name_mode (
    void
);

STRTAB_str_t   STRTAB_add_string(
    const char *string
);

void  STRTAB_str_to_string(
    STRTAB_str_t str,
    const char **strp
);

void  NAMETABLE_init(
    void
);

#ifdef DUMPERS
void  NAMETABLE_dump_tab(
    void
);

#endif

NAMETABLE_id_t NAMETABLE_add_derived_name(
    NAMETABLE_id_t id,
    const char *matrix
);

NAMETABLE_id_t NAMETABLE_add_derived_name2(
    NAMETABLE_id_t id1,
    NAMETABLE_id_t id2,
    char *matrix
);

void NAMETABLE_delete_node(
 NAMETABLE_id_t node
);

#endif
