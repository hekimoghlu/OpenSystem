/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
**      dgccallt.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  DG protocol service routines
**
**
*/

#ifndef _DGCCALLT_H
#define _DGCCALLT_H	1

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__dg_ccallt_insert (
        rpc_dg_ccall_p_t /*ccall*/
    );

PRIVATE void rpc__dg_ccallt_remove (
        rpc_dg_ccall_p_t /*ccall*/
    );

PRIVATE rpc_dg_ccall_p_t rpc__dg_ccallt_lookup (
        uuid_p_t /*actid*/,
        unsigned32 /*probe_hint*/
    );

PRIVATE void rpc__dg_ccallt_fork_handler (
        rpc_fork_stage_id_t /*stage*/
   );

#ifdef __cplusplus
}
#endif

#endif /* _DGCCALLT_H */
