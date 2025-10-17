/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
**      ndrglob.c
**
**  FACILITY:
**
**      Network Data Representation (NDR)
**
**  ABSTRACT:
**
**  Runtime global variable definitions.
**
**
*/

#include <commonp.h>
#include <ndrp.h>
#include <ndrglob.h>

GLOBAL u_char ndr_g_local_drep_packed[4] = {
    (NDR_LOCAL_INT_REP << 4) | NDR_LOCAL_CHAR_REP,
    NDR_LOCAL_FLOAT_REP,
    0,
    0,
};

GLOBAL
ndr_format_t ndr_g_local_drep = {
    NDR_LOCAL_INT_REP,
    NDR_LOCAL_CHAR_REP,
    NDR_LOCAL_FLOAT_REP,
    0
};

GLOBAL rpc_transfer_syntax_t ndr_g_transfer_syntax = {
    {
        {
            0x8a885d04U, 0x1ceb, 0x11c9, 0x9f, 0xe8,
            {0x8, 0x0, 0x2b, 0x10, 0x48, 0x60}
        },
        2
    },
    0,
    NULL
};
