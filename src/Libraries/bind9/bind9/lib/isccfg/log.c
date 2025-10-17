/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
/* $Id: log.c,v 1.11 2007/06/19 23:47:22 tbox Exp $ */

/*! \file */

#include <config.h>

#include <isc/util.h>

#include <isccfg/log.h>

/*%
 * When adding a new category, be sure to add the appropriate
 * \#define to <isccfg/log.h>.
 */
LIBISCCFG_EXTERNAL_DATA isc_logcategory_t cfg_categories[] = {
	{ "config", 	0 },
	{ NULL, 	0 }
};

/*%
 * When adding a new module, be sure to add the appropriate
 * \#define to <isccfg/log.h>.
 */
LIBISCCFG_EXTERNAL_DATA isc_logmodule_t cfg_modules[] = {
	{ "isccfg/parser",	0 },
	{ NULL, 		0 }
};

void
cfg_log_init(isc_log_t *lctx) {
	REQUIRE(lctx != NULL);

	isc_log_registercategories(lctx, cfg_categories);
	isc_log_registermodules(lctx, cfg_modules);
}
