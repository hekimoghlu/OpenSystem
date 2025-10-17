/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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
**      ifspec.h
**
**  FACILITY:
**
**      IDL Compiler Backend
**
**  ABSTRACT:
**
**  Header file for ifspec.c
**
**  VERSION: DCE 1.0
**
*/

#ifndef IFSPEC_H
#define IFSPEC_H

extern void CSPELL_interface_def(
    FILE *fid,
    AST_interface_n_t *ifp,
    BE_output_k_t kind,
    boolean generate_mepv
);

void CSPELL_manager_epv
(
    FILE *fid,
    AST_interface_n_t *ifp
);

#endif
