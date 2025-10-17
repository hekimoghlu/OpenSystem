/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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
**      comsoc_bsd.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  The platform-specific portion of the internal network "socket" object
**  interface.  See abstract in "comsoc.h" for details.
**
**  For BSD 4.3 & 4.4 systems.
**
*/

#ifndef _COMSOC_BSD_H
#define _COMSOC_BSD_H	1

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include <comnaf.h>
#include <commonp.h>
#include <ipnaf.h>

PRIVATE rpc_socket_error_t
rpc__bsd_socket_open_basic(
    rpc_naf_id_t        naf,
    rpc_network_if_id_t net_if,
    rpc_network_protocol_id_t net_prot ATTRIBUTE_UNUSED,
    rpc_socket_basic_t        *sock
    );

PRIVATE rpc_socket_error_t
rpc__bsd_socket_close_basic(
    rpc_socket_basic_t        sock
    );

EXTERNAL PRIVATE const rpc_socket_vtbl_t rpc_g_bsd_socket_vtbl;

#endif /* _COMSOC_BSD_H */
