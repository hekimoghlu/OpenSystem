/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
/* $Id: tsigconf.h,v 1.18 2009/06/11 23:47:55 tbox Exp $ */

#ifndef NS_TSIGCONF_H
#define NS_TSIGCONF_H 1

/*! \file */

#include <isc/types.h>
#include <isc/lang.h>

ISC_LANG_BEGINDECLS

isc_result_t
ns_tsigkeyring_fromconfig(const cfg_obj_t *config, const cfg_obj_t *vconfig,
			  isc_mem_t *mctx, dns_tsig_keyring_t **ringp);
/*%<
 * Create a TSIG key ring and configure it according to the 'key'
 * statements in the global and view configuration objects.
 *
 *	Requires:
 *	\li	'config' is not NULL.
 *	\li	'vconfig' is not NULL.
 *	\li	'mctx' is not NULL
 *	\li	'ringp' is not NULL, and '*ringp' is NULL
 *
 *	Returns:
 *	\li	ISC_R_SUCCESS
 *	\li	ISC_R_NOMEMORY
 */

ISC_LANG_ENDDECLS

#endif /* NS_TSIGCONF_H */
