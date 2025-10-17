/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
**      rpcrand.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  The support routines for the rpcrand.h abstraction.  These should NOT
**  be called directly; use the macros defined in rpcrand.h .
**
**
*/

#include <commonp.h>

/*
 * R P C _ _ R A N D O M _ I N I T
 */

PRIVATE void rpc__random_init
(
    unsigned seed
)
{
    srandom ((unsigned) seed);
}

/*
 * R P C _ _ R A N D O M _ G E T
 */

PRIVATE unsigned32 rpc__random_get
(
    unsigned32 lower ATTRIBUTE_UNUSED,
    unsigned32 upper ATTRIBUTE_UNUSED
)
{
    return ( (unsigned32) (random () % UINT_MAX));
}
