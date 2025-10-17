/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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
**      cominitp.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Private interface to the Common Communications Service Initialization
**  Service.
**
**
*/

#ifndef _COMINITP_H
#define _COMINITP_H

/***********************************************************************/
/*
 * Note: these are defined for later use of shared images
 */

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__load_modules(void);

PRIVATE rpc_naf_init_fn_t rpc__load_naf (
        rpc_naf_id_elt_p_t              /*naf*/,
        unsigned32                      * /*st*/
    );

PRIVATE rpc_prot_init_fn_t rpc__load_prot (
        rpc_protocol_id_elt_p_t         /*rpc_protocol*/,
        unsigned32                      * /*st*/
    );

PRIVATE rpc_auth_init_fn_t rpc__load_auth (
        rpc_authn_protocol_id_elt_p_t   /*auth_protocol*/,
        unsigned32                      * /*st*/
    );

#ifdef __cplusplus
}
#endif

#endif /* _COMINITP_H */
