/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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
**      namtbpvt.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  This header file contains the private definitions necessary for the
**  nametable modules.
**
**  VERSION: DCE 1.0
**
*/
/********************************************************************/
/*                                                                  */
/*   NAMTBPVT.H                                                     */
/*                                                                  */
/*              Data types private to the nametable routines.       */
/*                                                                  */
/********************************************************************/

typedef struct NAMETABLE_binding_n_t {
        int                              bindingLevel;
        const void                      *theBinding;
        struct NAMETABLE_binding_n_t    *nextBindingThisLevel;
        struct NAMETABLE_binding_n_t    *oldBinding;
        NAMETABLE_id_t                   boundBy;
}
NAMETABLE_binding_n_t;

typedef struct NAMETABLE_n_t {
        struct NAMETABLE_n_t    *left;  /* Subtree with names less          */
        struct NAMETABLE_n_t    *right; /* Subtree with names greater       */
        struct NAMETABLE_n_t    *parent;/* Parent in the tree               */
                                        /* NULL if this is the root         */
        const char              *id;    /* The identifier string            */
        NAMETABLE_binding_n_t   *bindings;      /* The list of bindings known       */
                                                /* by this name at this time.       */
        NAMETABLE_binding_n_t   *tagBinding;    /* The structure known by this tag. */
}
NAMETABLE_n_t;

typedef struct NAMETABLE_temp_name_t {
        struct NAMETABLE_temp_name_t * next;  /* Next temp name chain block */
        NAMETABLE_id_t   node;                /* The temp name tree node    */
}
NAMETABLE_temp_name_t;
