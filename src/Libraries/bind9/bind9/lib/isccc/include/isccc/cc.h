/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
/* $Id: cc.h,v 1.11 2007/08/28 07:20:43 tbox Exp $ */

#ifndef ISCCC_CC_H
#define ISCCC_CC_H 1

/*! \file isccc/cc.h */

#include <isc/lang.h>
#include <isccc/types.h>

ISC_LANG_BEGINDECLS

/*% from lib/dns/include/dst/dst.h */

#define ISCCC_ALG_UNKNOWN	0
#define ISCCC_ALG_HMACMD5	157
#define ISCCC_ALG_HMACSHA1	161
#define ISCCC_ALG_HMACSHA224	162
#define ISCCC_ALG_HMACSHA256	163
#define ISCCC_ALG_HMACSHA384	164
#define ISCCC_ALG_HMACSHA512	165

/*% Maximum Datagram Package */
#define ISCCC_CC_MAXDGRAMPACKET		4096

/*% Message Type String */
#define ISCCC_CCMSGTYPE_STRING		0x00
/*% Message Type Binary Data */
#define ISCCC_CCMSGTYPE_BINARYDATA	0x01
/*% Message Type Table */
#define ISCCC_CCMSGTYPE_TABLE		0x02
/*% Message Type List */
#define ISCCC_CCMSGTYPE_LIST		0x03

/*% Send to Wire */
isc_result_t
isccc_cc_towire(isccc_sexpr_t *alist, isccc_region_t *target,
		isc_uint32_t algorithm, isccc_region_t *secret);

/*% Get From Wire */
isc_result_t
isccc_cc_fromwire(isccc_region_t *source, isccc_sexpr_t **alistp,
		  isc_uint32_t algorithm, isccc_region_t *secret);

/*% Create Message */
isc_result_t
isccc_cc_createmessage(isc_uint32_t version, const char *from, const char *to,
		       isc_uint32_t serial, isccc_time_t now,
		       isccc_time_t expires, isccc_sexpr_t **alistp);

/*% Create Acknowledgment */
isc_result_t
isccc_cc_createack(isccc_sexpr_t *message, isc_boolean_t ok,
		   isccc_sexpr_t **ackp);

/*% Is Ack? */
isc_boolean_t
isccc_cc_isack(isccc_sexpr_t *message);

/*% Is Reply? */
isc_boolean_t
isccc_cc_isreply(isccc_sexpr_t *message);

/*% Create Response */
isc_result_t
isccc_cc_createresponse(isccc_sexpr_t *message, isccc_time_t now,
			isccc_time_t expires, isccc_sexpr_t **alistp);

/*% Define String */
isccc_sexpr_t *
isccc_cc_definestring(isccc_sexpr_t *alist, const char *key, const char *str);

/*% Define uint 32 */
isccc_sexpr_t *
isccc_cc_defineuint32(isccc_sexpr_t *alist, const char *key, isc_uint32_t i);

/*% Lookup String */
isc_result_t
isccc_cc_lookupstring(isccc_sexpr_t *alist, const char *key, char **strp);

/*% Lookup uint 32 */
isc_result_t
isccc_cc_lookupuint32(isccc_sexpr_t *alist, const char *key,
		      isc_uint32_t *uintp);

/*% Create Symbol Table */
isc_result_t
isccc_cc_createsymtab(isccc_symtab_t **symtabp);

/*% Clean up Symbol Table */
void
isccc_cc_cleansymtab(isccc_symtab_t *symtab, isccc_time_t now);

/*% Check for Duplicates */
isc_result_t
isccc_cc_checkdup(isccc_symtab_t *symtab, isccc_sexpr_t *message,
		  isccc_time_t now);

ISC_LANG_ENDDECLS

#endif /* ISCCC_CC_H */
