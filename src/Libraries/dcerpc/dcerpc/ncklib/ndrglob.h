/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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
**      ndrglob.h
**
**  FACILITY:
**
**      Network Data Representation (NDR)
**
**  ABSTRACT:
**
**  Runtime global variable (external) declarations.
**
**
*/

#ifndef _NDRGLOB_H
#define _NDRGLOB_H 1

#include <sys/types.h>

    /*
     * Local data representation.
     *
     * "ndr_g_local_drep" is what stubs use when they're interested in
     * the local data rep.  "ndr_local_drep_packed" is the actual correct
     * 4-byte wire format of the local drep, suitable for copying into
     * packet headers.
     */

EXTERNAL u_char ndr_g_local_drep_packed[4];

EXTERNAL ndr_format_t ndr_g_local_drep;

    /*
     * A constant transfer syntax descriptor that says "NDR".
     */

EXTERNAL rpc_transfer_syntax_t ndr_g_transfer_syntax;

#endif /* _NDRGLOB_H */
