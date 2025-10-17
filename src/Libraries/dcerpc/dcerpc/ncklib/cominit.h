/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
**      cominit.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Interface to the Common Communications Service Initialization Service.
**
**
*/

#ifndef _COMINIT_H
#define _COMINIT_H

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************/
/*
 * R P C _ _ I N I T
 *
 */

PRIVATE void rpc__init ( void );

PRIVATE void rpc__fork_handler (
        rpc_fork_stage_id_t   /*stage*/

    );

PRIVATE void rpc__set_port_restriction_from_string (
        unsigned_char_p_t  /*input_string*/,
        unsigned32         * /*status*/
    );

    PRIVATE void rpc__static_init(void);

#ifdef __cplusplus
}
#endif

#endif /* _COMINIT_H */
