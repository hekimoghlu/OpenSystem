/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
#ifndef _GSSD_MACH_TYPES_H_
#define _GSSD_MACH_TYPES_H_

#define MAX_DISPLAY_STR 128
#define MAX_PRINC_STR 1024

typedef enum gssd_mechtype {
	GSSD_NO_MECH = -1,
	GSSD_KRB5_MECH = 0,
	GSSD_SPNEGO_MECH,
	GSSD_NTLM_MECH,
	GSSD_IAKERB_MECH
} gssd_mechtype;

typedef enum gssd_nametype {
	GSSD_STRING_NAME = 0,
	GSSD_EXPORT,
	GSSD_ANONYMOUS,
	GSSD_HOSTBASED,
	GSSD_USER,
	GSSD_MACHINE_UID,
	GSSD_STRING_UID,
	GSSD_KRB5_PRINCIPAL,
	GSSD_KRB5_REFERRAL,
	GSSD_NTLM_PRINCIPAL,
	GSSD_NTLM_BLOB,
	GSSD_UUID
} gssd_nametype;

typedef char *gssd_string;
typedef char *gssd_dstring;
typedef uint8_t *gssd_byte_buffer;
typedef uint32_t *gssd_gid_list;
typedef uint64_t gssd_ctx;
typedef uint64_t gssd_cred;
typedef int32_t *gssd_etype_list;

/* The following need to correspond to GSS_C_*_FLAG in gssapi.h */
#define GSSD_DELEG_FLAG         1
#define GSSD_MUTUAL_FLAG        2
#define GSSD_REPLAY_FLAG        4
#define GSSD_SEQUENCE_FLAG      8
#define GSSD_CONF_FLAG          16
#define GSSD_INTEG_FLAG         32
#define GSSD_ANON_FLAG          64
#define GSSD_PROT_FLAG          128
#define GSSD_TRANS_FLAG         256
#define GSSD_DELEG_POLICY_FLAG  32768

#define GSSD_NO_DEFAULT         1  // Only use the supplied principal, do not fallback to the default.
#define GSSD_NO_CANON           2  // Don't canononicalize host names
#define GSSD_HOME_ACCESS_OK     4  // OK to access home directory
#define GSSD_GUEST_ONLY         8  // NTLM Server is forcing guest access
#define GSSD_RESTART            16 // Destroy the supplied context and start over
#define GSSD_NFS_1DES           64 // Only get single DES session keys
#define GSSD_WIN2K_HACK         128 // Hack for Win2K
#define GSSD_LUCID_CONTEXT      256 // Export Lucid context

#endif /* _GSSD_MACH_TYPES_H_ */
