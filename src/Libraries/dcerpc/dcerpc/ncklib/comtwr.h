/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#ifndef _COMTWR_H
#define _COMTWR_H 1
/*
**
**
**  NAME:
**
**      comtwr.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**      Header file containing macros, definitions, typedefs and prototypes
**      of exported routines from the comtwr.c module.
**
**
**/

#ifdef __cplusplus
extern "C" {
#endif

PRIVATE void rpc__tower_free (
    twr_p_t                 * /*tower*/,
    unsigned32              * /*status*/ );

PRIVATE void rpc__tower_from_tower_ref (
    rpc_tower_ref_p_t        /*tower_ref*/,
    twr_p_t                 * /*tower*/,
    unsigned32              * /*status*/ );

PRIVATE void rpc__tower_to_tower_ref (
   twr_p_t                  /*tower*/,
   rpc_tower_ref_p_t       * /*tower_ref*/,
   unsigned32              * /*status*/ );

#endif /* _COMTRW_H */
