/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
**      cspeldcl.h
**
**  FACILITY:
**
**      IDL Compiler Backend
**
**  ABSTRACT:
**
**  Header file for cspeldcl.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef CSPELDCL_H
#define CSPELDCL_H

extern void CSPELL_constant_val (
    FILE *fid, AST_constant_n_t *cp
);

extern void CSPELL_labels (
    FILE *fid, AST_case_label_n_t *tgp
);

extern void CSPELL_parameter_list (
    FILE        *fid,
    AST_parameter_n_t *pp,
    boolean encoding_services
);

extern void CSPELL_finish_synopsis (
    FILE *fid, AST_parameter_n_t *paramlist
);

#endif
