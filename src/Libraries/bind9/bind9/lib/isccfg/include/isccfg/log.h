/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
/* $Id: log.h,v 1.14 2009/01/18 23:48:14 tbox Exp $ */

#ifndef ISCCFG_LOG_H
#define ISCCFG_LOG_H 1

/*! \file isccfg/log.h */

#include <isc/lang.h>
#include <isc/log.h>

LIBISCCFG_EXTERNAL_DATA extern isc_logcategory_t cfg_categories[];
LIBISCCFG_EXTERNAL_DATA extern isc_logmodule_t cfg_modules[];

#define CFG_LOGCATEGORY_CONFIG	(&cfg_categories[0])

#define CFG_LOGMODULE_PARSER	(&cfg_modules[0])

ISC_LANG_BEGINDECLS

void
cfg_log_init(isc_log_t *lctx);
/*%<
 * Make the libisccfg categories and modules available for use with the
 * ISC logging library.
 *
 * Requires:
 *\li	lctx is a valid logging context.
 *
 *\li	cfg_log_init() is called only once.
 *
 * Ensures:
 * \li	The categories and modules defined above are available for
 * 	use by isc_log_usechannnel() and isc_log_write().
 */

ISC_LANG_ENDDECLS

#endif /* ISCCFG_LOG_H */
