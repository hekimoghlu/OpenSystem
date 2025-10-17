/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
/*! \file dns/log.h
 * \author  Principal Authors: DCL */

#ifndef DNS_LOG_H
#define DNS_LOG_H 1

#include <isc/lang.h>
#include <isc/log.h>

LIBDNS_EXTERNAL_DATA extern isc_log_t *dns_lctx;
LIBDNS_EXTERNAL_DATA extern isc_logcategory_t dns_categories[];
LIBDNS_EXTERNAL_DATA extern isc_logmodule_t dns_modules[];

#define DNS_LOGCATEGORY_NOTIFY		(&dns_categories[0])
#define DNS_LOGCATEGORY_DATABASE	(&dns_categories[1])
#define DNS_LOGCATEGORY_SECURITY	(&dns_categories[2])
/* DNS_LOGCATEGORY_CONFIG superseded by CFG_LOGCATEGORY_CONFIG */
#define DNS_LOGCATEGORY_DNSSEC		(&dns_categories[4])
#define DNS_LOGCATEGORY_RESOLVER	(&dns_categories[5])
#define DNS_LOGCATEGORY_XFER_IN		(&dns_categories[6])
#define DNS_LOGCATEGORY_XFER_OUT	(&dns_categories[7])
#define DNS_LOGCATEGORY_DISPATCH	(&dns_categories[8])
#define DNS_LOGCATEGORY_LAME_SERVERS	(&dns_categories[9])
#define DNS_LOGCATEGORY_DELEGATION_ONLY	(&dns_categories[10])
#define DNS_LOGCATEGORY_EDNS_DISABLED	(&dns_categories[11])
#define DNS_LOGCATEGORY_RPZ		(&dns_categories[12])
#define DNS_LOGCATEGORY_RRL		(&dns_categories[13])
#define DNS_LOGCATEGORY_CNAME		(&dns_categories[14])
#define DNS_LOGCATEGORY_SPILL		(&dns_categories[15])

/* Backwards compatibility. */
#define DNS_LOGCATEGORY_GENERAL		ISC_LOGCATEGORY_GENERAL

#define DNS_LOGMODULE_DB		(&dns_modules[0])
#define DNS_LOGMODULE_RBTDB		(&dns_modules[1])
#define DNS_LOGMODULE_RBTDB64		(&dns_modules[2])
#define DNS_LOGMODULE_RBT		(&dns_modules[3])
#define DNS_LOGMODULE_RDATA		(&dns_modules[4])
#define DNS_LOGMODULE_MASTER		(&dns_modules[5])
#define DNS_LOGMODULE_MESSAGE		(&dns_modules[6])
#define DNS_LOGMODULE_CACHE		(&dns_modules[7])
#define DNS_LOGMODULE_CONFIG		(&dns_modules[8])
#define DNS_LOGMODULE_RESOLVER		(&dns_modules[9])
#define DNS_LOGMODULE_ZONE		(&dns_modules[10])
#define DNS_LOGMODULE_JOURNAL		(&dns_modules[11])
#define DNS_LOGMODULE_ADB		(&dns_modules[12])
#define DNS_LOGMODULE_XFER_IN		(&dns_modules[13])
#define DNS_LOGMODULE_XFER_OUT		(&dns_modules[14])
#define DNS_LOGMODULE_ACL		(&dns_modules[15])
#define DNS_LOGMODULE_VALIDATOR		(&dns_modules[16])
#define DNS_LOGMODULE_DISPATCH		(&dns_modules[17])
#define DNS_LOGMODULE_REQUEST		(&dns_modules[18])
#define DNS_LOGMODULE_MASTERDUMP	(&dns_modules[19])
#define DNS_LOGMODULE_TSIG		(&dns_modules[20])
#define DNS_LOGMODULE_TKEY		(&dns_modules[21])
#define DNS_LOGMODULE_SDB		(&dns_modules[22])
#define DNS_LOGMODULE_DIFF		(&dns_modules[23])
#define DNS_LOGMODULE_HINTS		(&dns_modules[24])
#define DNS_LOGMODULE_ACACHE		(&dns_modules[25])
#define DNS_LOGMODULE_DLZ		(&dns_modules[26])
#define DNS_LOGMODULE_DNSSEC		(&dns_modules[27])
#define DNS_LOGMODULE_CRYPTO		(&dns_modules[28])
#define DNS_LOGMODULE_PACKETS		(&dns_modules[29])

ISC_LANG_BEGINDECLS

void
dns_log_init(isc_log_t *lctx);
/*%
 * Make the libdns categories and modules available for use with the
 * ISC logging library.
 *
 * Requires:
 *\li	lctx is a valid logging context.
 *
 *\li	dns_log_init() is called only once.
 *
 * Ensures:
 * \li	The categories and modules defined above are available for
 * 	use by isc_log_usechannnel() and isc_log_write().
 */

void
dns_log_setcontext(isc_log_t *lctx);
/*%
 * Make the libdns library use the provided context for logging internal
 * messages.
 *
 * Requires:
 *\li	lctx is a valid logging context.
 */

ISC_LANG_ENDDECLS

#endif /* DNS_LOG_H */
