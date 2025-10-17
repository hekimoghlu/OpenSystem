/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
**      ndrcharp.c
**
**  FACILITY:
**
**      IDL Stub Runtime Support
**
**  ABSTRACT:
**
**  This module contains the declarations of two character table pointers
**  used by the stub support library for translating to/from ascii/ebcdic.
**
**  VERSION: DCE 1.0
**
*/

#if HAVE_CONFIG_H
#include <config.h>
#endif

/* The ordering of the following 3 includes should NOT be changed! */
#include <dce/rpc.h>
#include <dce/stubbase.h>
#include <lsysdep.h>

/*
 * Pointer cells for the two tables.
 */
globaldef rpc_trans_tab_p_t ndr_g_ascii_to_ebcdic = (rpc_trans_tab_p_t) "NDR_G_ASCII_TO_EBCDIC";
globaldef rpc_trans_tab_p_t ndr_g_ebcdic_to_ascii = (rpc_trans_tab_p_t) "NDR_G_EBCDIC_TO_ASCII";
