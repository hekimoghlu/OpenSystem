/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#ifndef SecureNetworking_ipsec_xpc_H
#define SecureNetworking_ipsec_xpc_H

#define SN_ENTITLEMENT_IPSEC_IKE    CFSTR("com.apple.private.SecureNetworking.ipsec_ike")
#define SN_ENTITLEMENT_IPSEC_DB     CFSTR("com.apple.private.SecureNetworking.ipsec_db")

#define IPSEC_HELPER    "com.apple.SecureNetworking.IPSec"

/* IKE */
#define	IPSECOPCODE         "ipsecopcode"
#define	IPSECOPIKEDICT      "ipsecikedict"
#define	IPSECOPCHILDDICT    "ipsecchilddict"
#define	IPSECOBJREF         "ipsecobjectref"
#define IPSECIKEID          "ipsecikeid"
#define IPSECCHILDID        "ipsecchildid"
#define IPSECIKESTATUS      "ipsecikestatus"
#define IPSECCHILDSTATUS    "ipsecchildstatus"


/* DB SA */
#define IPSECSASESSIONID    "ipsecsasessionid"
#define IPSECSAID           "ipsecsaid"
#define IPSECSADICT         "ipsecsadict"
#define IPSECSASPI          "ipsecsaspi"
#define IPSECSAIDARRAY      "ipsecsaidarray"
#define IPSECPOLICYID       "ipsecpolicyid"
#define	IPSECPOLICYDICT     "ipsecpolicydict"
#define IPSECPOLICYIDARRAY  "ipsecpolicyidarray"

/* message */
#define IPSECMESSAGE        "ipsecmessage"
#define IPSECITEMID         "ipsecitemid"
#define IPSECITEMDICT       "ipsecitemdict"

#define SERVERREPLY         "reply"

#define REPLYOFFSET         0x1000

enum {
	IPSECIKE_CREATE         = 0x0001,
	IPSECIKE_START,
	IPSECIKE_STOP,
	IPSECIKE_GETSTATUS,
	IPSECIKE_INVALIDATE,
	IPSECIKE_START_CHILD,
	IPSECIKE_STOP_CHILD,
	IPSECIKE_ENABLE_CHILD,
	IPSECIKE_DISABLE_CHILD,
	IPSECIKE_GETSTATUS_CHILD
};


enum {
    IPSECDB_CREATESESSION  = 0x0101,
	IPSECDB_GETSPI,
	IPSECDB_ADDSA,
    IPSECDB_UPDATESA,
	IPSECDB_DELETESA,
	IPSECDB_COPYSA,
	IPSECDB_FLUSHSA,
	IPSECDB_ADDPOLICY,
	IPSECDB_DELETEPOLICY,
	IPSECDB_COPYPOLICY,
	IPSECDB_FLUSHPOLICIES,
	IPSECDB_FLUSHALL,
	IPSECDB_INVALIDATE,
    IPSECDB_COPYSAIDS,
    IPSECDB_COPYPOLICYIDS
};

enum {
	SERVER_REPLY_OK		= 0x0000,
	SERVER_FAILED
};

#endif
