/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
**      cnid.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Interface to the Local Identifier Service.
**
**
*/

#ifndef _CNID_H
#define _CNID_H	1

/*
 * R P C _ C N _ L O C A L _ I D _ E Q U A L
 */
#define RPC_CN_LOCAL_ID_EQUAL(id1, id2)\
    ((id1.parts.id_seqnum == id2.parts.id_seqnum) &&\
     (id1.parts.id_index == id2.parts.id_index))

/*
 * R P C _ C N _ L O C A L _ I D _ V A L I D
 */

#define RPC_CN_LOCAL_ID_VALID(id) (id.parts.id_seqnum != 0)

/*
 * R P C _ C N _ L O C A L _ I D _ C L E A R
 */
#define RPC_CN_LOCAL_ID_CLEAR(id)\
{\
    id.parts.id_seqnum = 0;\
    id.parts.id_index = 0;\
}

/*
 * R P C _ _ C N _ I N I T _ S E Q N U M
 *
 * This routine initializes the global sequence number cell and
 * corresponding mutex.
 */

void rpc__cn_init_seqnum (void);

/*
 * R P C _ _ C N _ G E N _ L O C A L _ I D
 *
 * This routine creates a new local identifier.
 */

void rpc__cn_gen_local_id (
    unsigned32          /* index */,
    rpc_cn_local_id_t   * /* lcl_id */);

#endif /* _CNID_H */
