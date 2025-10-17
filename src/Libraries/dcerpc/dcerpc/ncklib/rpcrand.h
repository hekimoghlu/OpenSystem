/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
**      rpcrand.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Random number generator abstraction to isolate random number generation
**  routines and allow alternate implementations to be substituted more
**  easily.
**
**  This package provides the following PRIVATE operations:
**
**      void       RPC_RANDOM_INIT(seed)
**      unsigned32 RPC_RANDOM_GET(lower, upper)
**
**
*/

#ifndef _RPCRAND_H
#define _RPCRAND_H

/*
 * R P C _ R A N D O M _ I N I T
 *
 * Used for random number 'seed' routines or any other one time
 * initialization required.
 */

#define RPC_RANDOM_INIT(seed) \
        rpc__random_init(seed)

/*
 * R P C _ R A N D O M _ G E T
 *
 * Get a random number in the range lower - upper (inclusive)
 */

#define RPC_RANDOM_GET(lower, upper) \
        (((rpc__random_get(lower, upper)) % ((upper - lower + 1) != 0 ? (upper - lower + 1) : 1)     ) + lower)

/*
 * Prototype for the private 'c' routines used by the RPC_RANDOM macros.
 */

PRIVATE void rpc__random_init ( unsigned /*seed*/ );

PRIVATE unsigned32 rpc__random_get (
        unsigned32  /*lower*/,
        unsigned32  /*upper*/
    );

#endif /* _RPCRAND_H */
