/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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
**      comcthd.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Definitions of types/constants for the Call Thread Services
**  of the Common Communications Service component of the RPC runtime.
**
**
*/

#ifndef _COMCTHD_H
#define _COMCTHD_H	1

#ifdef _cplusplus
extern "C" {
#endif

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ I N I T
 *
 */

PRIVATE void rpc__cthread_init (
        unsigned32                  * /*status*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ S T A R T _ A L L
 *
 */

PRIVATE void rpc__cthread_start_all (
        unsigned32              /*default_pool_cthreads*/,
        unsigned32              * /*status*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ S T O P _ A L L
 *
 */

PRIVATE void rpc__cthread_stop_all (
        unsigned32              * /*status*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ I N V O K E _ N U L L
 *
 */

PRIVATE void rpc__cthread_invoke_null (
        rpc_call_rep_p_t        /*call_rep*/,
        uuid_p_t                /*object*/,
        uuid_p_t                /*if_uuid*/,
        unsigned32              /*if_ver*/,
        unsigned32              /*if_opnum*/,
        rpc_prot_cthread_executor_fn_t /*cthread_executor*/,
        dce_pointer_t               /*call_args*/,
        unsigned32              * /*status*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ D E Q U E U E
 *
 */

PRIVATE boolean32 rpc__cthread_dequeue (
        rpc_call_rep_p_t        /*call*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ C A N C E L
 *
 */

PRIVATE void rpc__cthread_cancel (
        rpc_call_rep_p_t        /*call*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ C A N C E L _ C A F
 *
 */

PRIVATE boolean32 rpc__cthread_cancel_caf (
        rpc_call_rep_p_t        /*call*/
    );

/***********************************************************************/
/*
 * R P C _ _ C T H R E A D _ C A N C E L _ E N A B L E _ P O S T I N G
 *
 */
 PRIVATE void rpc__cthread_cancel_enable_post (
        rpc_call_rep_p_t        /*call*/
    );

#ifdef _cplusplus
}
#endif

#endif /* _COMCTHD_H */
