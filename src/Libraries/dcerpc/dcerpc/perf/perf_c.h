/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
**      perf_c.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Common header file used by perf client and server.
**
**
*/

#include <signal.h>
#include <stdio.h>
#include <math.h>

#include <dce/nbase.h>
#include <dce/rpc.h>

#include <perf.h>
#include <perfb.h>
#include <perfc.h>
#include <perfg.h>

#ifndef NO_TASKING
#  include <dce/pthread.h>
#  include <dce/exc_handling.h>
#  ifdef BROKEN_CMA_EXC_HANDLING
#    define pthread_cancel_e  cma_e_alerted
#  endif
#endif

#include <dce/rpcexc.h>

#if defined(SYS5)
#  define index strchr
#endif

extern char *error_text();

#define MARSHALL_DOUBLE(d)
#define UNMARSHALL_DOUBLE(d)

extern uuid_old_t FooType, BarType, FooObj1, FooObj2, BarObj1, BarObj2;
extern idl_uuid_t NilTypeObj, NilObj, ZotObj, ZotType;

extern char *authn_level_names[];
extern char *authn_names[];
extern char *authz_names[];

#define DEBUG_LEVEL   "0.1"
#define LOSSY_LEVEL   "4.99"

#ifdef CMA_INCLUDE
#define USE_PTHREAD_DELAY_NP
#endif

#ifdef USE_PTHREAD_DELAY_NP

#define SLEEP(secs) \
{ \
    struct timespec delay; \
    delay.tv_sec  = (secs); \
    delay.tv_nsec = 0; \
    pthread_delay_np(&delay); \
}

#else

#define SLEEP(secs) \
    sleep(secs)

#endif

#define VRprintf(level, stuff) \
{ \
    if (verbose >= (level)) \
        printf stuff; \
}
