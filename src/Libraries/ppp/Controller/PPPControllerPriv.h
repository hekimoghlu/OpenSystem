/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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
#ifndef __PPPCONTROLLERPRIV_H__
#define __PPPCONTROLLERPRIV_H__

/*
 * Keys have moved to SystemConfiguration Framework
 *  SCSchemaDefinitions.h and SCSchemaDefinitionsPrivate.h 
 */

/* IPSec error codes */
enum {
	IPSEC_NO_ERROR = 0,
	IPSEC_GENERIC_ERROR = 1,
	IPSEC_NOSERVERADDRESS_ERROR = 2,
	IPSEC_NOSHAREDSECRET_ERROR = 3,
	IPSEC_NOCERTIFICATE_ERROR = 4,
	IPSEC_RESOLVEADDRESS_ERROR = 5,
	IPSEC_NOLOCALNETWORK_ERROR = 6,
	IPSEC_CONFIGURATION_ERROR = 7,
	IPSEC_RACOONCONTROL_ERROR = 8,
	IPSEC_CONNECTION_ERROR = 9,
	IPSEC_NEGOTIATION_ERROR = 10,
	IPSEC_SHAREDSECRET_ERROR = 11,
	IPSEC_SERVER_CERTIFICATE_ERROR = 12,
	IPSEC_CLIENT_CERTIFICATE_ERROR = 13,
	IPSEC_XAUTH_ERROR = 14,
	IPSEC_NETWORKCHANGE_ERROR = 15,
	IPSEC_PEERDISCONNECT_ERROR = 16,
	IPSEC_PEERDEADETECTION_ERROR = 17,
	IPSEC_EDGE_ACTIVATION_ERROR = 18,
	IPSEC_IDLETIMEOUT_ERROR = 19,
	IPSEC_CLIENT_CERTIFICATE_PREMATURE = 20,
	IPSEC_CLIENT_CERTIFICATE_EXPIRED = 21,
	IPSEC_SERVER_CERTIFICATE_PREMATURE = 22,
	IPSEC_SERVER_CERTIFICATE_EXPIRED = 23,
	IPSEC_SERVER_CERTIFICATE_INVALID_ID = 24
};

/* 
 * IPSEC state
 *
 */
enum {
    IPSEC_IDLE = 0,
    IPSEC_INITIALIZE,
    IPSEC_CONTACT,
    IPSEC_PHASE1,
    IPSEC_PHASE1AUTH,
    IPSEC_PHASE2,
    IPSEC_RUNNING,
    IPSEC_TERMINATE,
    IPSEC_WAITING
};



#endif


