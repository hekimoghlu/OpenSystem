/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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
/* $Id: log.h,v 1.27 2009/01/07 23:47:46 tbox Exp $ */

#ifndef NAMED_LOG_H
#define NAMED_LOG_H 1

/*! \file */

#include <isc/log.h>
#include <isc/types.h>

#include <dns/log.h>

#include <named/globals.h>	/* Required for ns_g_(categories|modules). */

/* Unused slot 0. */
#define NS_LOGCATEGORY_CLIENT		(&ns_g_categories[1])
#define NS_LOGCATEGORY_NETWORK		(&ns_g_categories[2])
#define NS_LOGCATEGORY_UPDATE		(&ns_g_categories[3])
#define NS_LOGCATEGORY_QUERIES		(&ns_g_categories[4])
#define NS_LOGCATEGORY_UNMATCHED	(&ns_g_categories[5])
#define NS_LOGCATEGORY_UPDATE_SECURITY	(&ns_g_categories[6])
#define NS_LOGCATEGORY_QUERY_ERRORS	(&ns_g_categories[7])

/*
 * Backwards compatibility.
 */
#define NS_LOGCATEGORY_GENERAL		ISC_LOGCATEGORY_GENERAL

#define NS_LOGMODULE_MAIN		(&ns_g_modules[0])
#define NS_LOGMODULE_CLIENT		(&ns_g_modules[1])
#define NS_LOGMODULE_SERVER		(&ns_g_modules[2])
#define NS_LOGMODULE_QUERY		(&ns_g_modules[3])
#define NS_LOGMODULE_INTERFACEMGR	(&ns_g_modules[4])
#define NS_LOGMODULE_UPDATE		(&ns_g_modules[5])
#define NS_LOGMODULE_XFER_IN		(&ns_g_modules[6])
#define NS_LOGMODULE_XFER_OUT		(&ns_g_modules[7])
#define NS_LOGMODULE_NOTIFY		(&ns_g_modules[8])
#define NS_LOGMODULE_CONTROL		(&ns_g_modules[9])
#define NS_LOGMODULE_LWRESD		(&ns_g_modules[10])

isc_result_t
ns_log_init(isc_boolean_t safe);
/*%
 * Initialize the logging system and set up an initial default
 * logging default configuration that will be used until the
 * config file has been read.
 *
 * If 'safe' is true, use a default configuration that refrains
 * from opening files.  This is to avoid creating log files
 * as root.
 */

isc_result_t
ns_log_setdefaultchannels(isc_logconfig_t *lcfg);
/*%
 * Set up logging channels according to the named defaults, which
 * may differ from the logging library defaults.  Currently,
 * this just means setting up default_debug.
 */

isc_result_t
ns_log_setsafechannels(isc_logconfig_t *lcfg);
/*%
 * Like ns_log_setdefaultchannels(), but omits any logging to files.
 */

isc_result_t
ns_log_setdefaultcategory(isc_logconfig_t *lcfg);
/*%
 * Set up "category default" to go to the right places.
 */

isc_result_t
ns_log_setunmatchedcategory(isc_logconfig_t *lcfg);
/*%
 * Set up "category unmatched" to go to the right places.
 */

void
ns_log_shutdown(void);

#endif /* NAMED_LOG_H */
