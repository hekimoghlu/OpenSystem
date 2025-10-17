/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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
/* $Id: check.h,v 1.9 2007/06/19 23:47:16 tbox Exp $ */

#ifndef BIND9_CHECK_H
#define BIND9_CHECK_H 1

/*! \file bind9/check.h */

#include <isc/lang.h>
#include <isc/types.h>

#include <isccfg/cfg.h>

ISC_LANG_BEGINDECLS

isc_result_t
bind9_check_namedconf(const cfg_obj_t *config, isc_log_t *logctx,
		      isc_mem_t *mctx);
/*%<
 * Check the syntactic validity of a configuration parse tree generated from
 * a named.conf file.
 *
 * Requires:
 *\li	config is a valid parse tree
 *
 *\li	logctx is a valid logging context.
 *
 * Returns:
 * \li	#ISC_R_SUCCESS
 * \li	#ISC_R_FAILURE
 */

isc_result_t
bind9_check_key(const cfg_obj_t *config, isc_log_t *logctx);
/*%<
 * Same as bind9_check_namedconf(), but for a single 'key' statement.
 */

ISC_LANG_ENDDECLS

#endif /* BIND9_CHECK_H */
