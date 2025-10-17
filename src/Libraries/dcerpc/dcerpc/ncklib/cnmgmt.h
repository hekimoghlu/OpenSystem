/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 23, 2024.
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
**      cnmgmt.h
**
**  FACILITY:
**
**      Remote Procedure Call (RPC)
**
**  ABSTRACT:
**
**  Interface to the NCA Connection Protocol Service's Management Service.
**
**
*/

#ifndef _CNMGMT_H
#define _CNMGMT_H	1

/*
 * R P C _ _ C N _ M G M T _ I N I T
 */

PRIVATE void rpc__cn_mgmt_init (void);

/*
 * R P C _ _ C N _ M G M T _ I N Q _ C A L L S _ S E N T
 */

PRIVATE unsigned32 rpc__cn_mgmt_inq_calls_sent (void);

/*
 * R P C _ _ C N _ M G M T _ I N Q _ C A L L S _ R C V D
 */

PRIVATE unsigned32 rpc__cn_mgmt_inq_calls_rcvd (void);

/*
 * R P C _ _ C N _ M G M T _ I N Q _ P K T S _ S E N T
 */

PRIVATE unsigned32 rpc__cn_mgmt_inq_pkts_sent (void);

/*
 * R P C _ _ C N _ M G M T _ I N Q _ P K T S _ R C V D
 */

PRIVATE unsigned32 rpc__cn_mgmt_inq_pkts_rcvd (void);

#endif /* _CNMGMT_H */
