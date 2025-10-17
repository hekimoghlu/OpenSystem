/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
/* $Id: zoneconf.h,v 1.30 2011/08/30 23:46:51 tbox Exp $ */

#ifndef NS_ZONECONF_H
#define NS_ZONECONF_H 1

/*! \file */

#include <isc/lang.h>
#include <isc/types.h>

#include <isccfg/aclconf.h>
#include <isccfg/cfg.h>

ISC_LANG_BEGINDECLS

isc_result_t
ns_zone_configure(const cfg_obj_t *config, const cfg_obj_t *vconfig,
		  const cfg_obj_t *zconfig, cfg_aclconfctx_t *ac,
		  dns_zone_t *zone, dns_zone_t *raw);
/*%<
 * Configure or reconfigure a zone according to the named.conf
 * data in 'cctx' and 'czone'.
 *
 * The zone origin is not configured, it is assumed to have been set
 * at zone creation time.
 *
 * Require:
 * \li	'lctx' to be initialized or NULL.
 * \li	'cctx' to be initialized or NULL.
 * \li	'ac' to point to an initialized ns_aclconfctx_t.
 * \li	'czone' to be initialized.
 * \li	'zone' to be initialized.
 */

isc_boolean_t
ns_zone_reusable(dns_zone_t *zone, const cfg_obj_t *zconfig);
/*%<
 * If 'zone' can be safely reconfigured according to the configuration
 * data in 'zconfig', return ISC_TRUE.  If the configuration data is so
 * different from the current zone state that the zone needs to be destroyed
 * and recreated, return ISC_FALSE.
 */


isc_result_t
ns_zone_configure_writeable_dlz(dns_dlzdb_t *dlzdatabase, dns_zone_t *zone,
				dns_rdataclass_t rdclass, dns_name_t *name);
/*%>
 * configure a DLZ zone, setting up the database methods and calling
 * postload to load the origin values
 *
 * Require:
 * \li	'dlzdatabase' to be a valid dlz database
 * \li	'zone' to be initialized.
 * \li	'rdclass' to be a valid rdataclass
 * \li	'name' to be a valid zone origin name
 */

ISC_LANG_ENDDECLS

#endif /* NS_ZONECONF_H */
