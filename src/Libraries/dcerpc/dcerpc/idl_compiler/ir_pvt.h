/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
**      ir_pvt.h
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  Header file containing defining intermediate-rep private data structures
**  for data that is kept in the IR_info_t field of some AST nodes.
**
**  %a%private_begin
**
**
**  %a%private_end
*/

#ifndef IR_PVTH_INCL
#define IR_PVTH_INCL

typedef struct IR_info_n_t {
    /*
     * Pointer to last created tuple in a parameter or type node's tuple list.
     */
    struct IR_tup_n_t   *cur_tup_p;
    /*
     * For a field, field number.  For other nodes, available for any use.
     */
    long                id_num;
    /*
     * On a param, T => requires server side preallocation of [ref] pointee(s).
     * On a type, T => same if reference is not under a full or unique pointer.
     */
    boolean             allocate_ref;
} IR_info_n_t;

typedef IR_info_n_t *IR_info_t;

/*
 * Data structure used to help sort the case labels of a union.
 */
typedef struct IR_case_info_n_t {
    struct AST_arm_n_t          *arm_p;     /* Ptr to union arm node */
    struct AST_case_label_n_t   *label_p;   /* Ptr to case label node */
    unsigned long               value;      /* Normallized case label value */
} IR_case_info_n_t;

#endif
