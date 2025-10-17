/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
**      rpcfork.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Various macros and data to assist with fork handling.
**
**
*/

#ifndef _RPCFORK_H
#define _RPCFORK_H	1

/*
 * Define constants to be passed to fork handling routines.  The value passed
 * indicates at which stage of the fork we are.
 */

#include <commonp.h>
#define RPC_C_PREFORK          1
#define RPC_C_POSTFORK_PARENT  2
#define RPC_C_POSTFORK_CHILD   3

typedef unsigned32       rpc_fork_stage_id_t;

PRIVATE void rpc__atfork ( void *handler);

#endif /* RCPFORK_H */
