/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
**      rpcp.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Server manager routines for performance and system execiser
**  auxiliary interface. This interface is generic.
**
**
*/

#include <perf_c.h>
#include <perf_p.h>

/***************************************************************************/

void foo_perfg_op1 (h, n, x)

handle_t                h __attribute__(unused);
unsigned long           n;
unsigned long           *x;

{
    *x = 2 * n;
}

/***************************************************************************/

void foo_perfg_op2 (h, n, x)

handle_t                h __attribute__(unused);
unsigned long           n;
unsigned long           *x;

{
    *x = 3 * n;
}

/***************************************************************************/

perfg_v1_0_epv_t foo_perfg_epv =
{
    foo_perfg_op1,
    foo_perfg_op2
};

/***************************************************************************/

void bar_perfg_op1 (h, n, x)

handle_t                h __attribute__(unused);
unsigned long           n;
unsigned long           *x;

{
    *x = 4 * n;
}

/***************************************************************************/

void bar_perfg_op2 (h, n, x)

handle_t                h __attribute__(unused);
unsigned long           n;
unsigned long           *x;

{
    *x = 5 * n;
}

/***************************************************************************/

perfg_v1_0_epv_t bar_perfg_epv =
{
    bar_perfg_op1,
    bar_perfg_op2
};
