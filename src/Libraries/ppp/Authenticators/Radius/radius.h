/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#ifndef __RADIUS_H__
#define __RADIUS_H__

#define MD4_SIGNATURE_SIZE	16	/* 16 bytes in a MD4 message digest */
#define MPPE_MAX_KEY_LEN        16      /* largest key length (128-bit) */


enum {
	RADIUS_USE_PAP		= 0x1,
	RADIUS_USE_CHAP		= 0x2,	// not supported
	RADIUS_USE_MSCHAP2	= 0x4,
	RADIUS_USE_EAP		= 0x8
};

struct auth_server {
	char	*address;
	char	*secret;
	int 	port;
	int 	timeout;
	int 	retries;
	u_int16_t 	proto;			// PAP/CHAP/MSCHAP2/EAP
};

// list of servers
extern struct auth_server **auth_servers;// array of authentication servers
extern int nb_auth_servers;				// number of authentication servers

// radius attributes
extern char		*nas_identifier;		// NAS Identifier to include in Radius packets
extern char		*nas_ip_address;		// NAS IP address to include in Radius packets
extern int		nas_port_type;			// default is virtual
extern int		tunnel_type;			// not specified

int radius_decryptmppekey(char *key, u_int8_t *attr_value, size_t attr_len, char *secret, char *authenticator, size_t auth_len);

int radius_eap_install(void);

#endif
