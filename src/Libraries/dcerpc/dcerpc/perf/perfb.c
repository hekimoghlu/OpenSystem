/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
**      perfb.c
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Server manager routines for performance and system execiser auxiliary
**  interface.  This interface is dynamically registered by the server when
**  request by the client through a call to an operation in the "perf"
**  interface.
**
**
*/

#include <perf_c.h>
#include <perf_p.h>
#include <unistd.h>

void print_binding_info(char *text, handle_t h);

perfb_v1_0_epv_t perfb_mgr_epv =
{
    perfb_init,
    perfb_in,
    perfb_brd,
    perfb_null,
    perfb_null_idem
};


/***************************************************************************/

void perfb_init
(
    handle_t                h,
    idl_char                *name
)
{
    print_binding_info ("perfb_init", h);
    gethostname(name, 256);
}

/***************************************************************************/

void perfb_in
(
    handle_t                h,
    perf_data_t             d,
    unsigned32              l,
    idl_boolean             verify,
    unsigned32           *sum
)
{
    print_binding_info ("perfb_in", h);
    perf_in(h, d, l, verify, sum);
}

/***************************************************************************/

void perfb_brd
(
    handle_t                h,
    idl_char                *name
)
{
    print_binding_info ("perfb_brd", h);
    gethostname(name, 256);
}

/***************************************************************************/

void perfb_null
(
    handle_t                h __attribute__(unused)
)
{
}

/***************************************************************************/

void perfb_null_idem
(
    handle_t                h __attribute__(unused)
)
{
}
