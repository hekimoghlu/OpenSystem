/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
/*! \file */

#include <config.h>

#include <isc/util.h>

#include <named/log.h>
#include <named/geoip.h>

#include <dns/geoip.h>

#ifdef HAVE_GEOIP
static dns_geoip_databases_t geoip_table = {
	NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};

static void
init_geoip_db(GeoIP **dbp, GeoIPDBTypes edition, GeoIPDBTypes fallback,
	      GeoIPOptions method, const char *name)
{
	char *info;
	GeoIP *db;

	REQUIRE(dbp != NULL);

	db = *dbp;

	if (db != NULL) {
		GeoIP_delete(db);
		db = *dbp = NULL;
	}

	if (! GeoIP_db_avail(edition)) {
		isc_log_write(ns_g_lctx, NS_LOGCATEGORY_GENERAL,
			NS_LOGMODULE_SERVER, ISC_LOG_INFO,
			"GeoIP %s (type %d) DB not available", name, edition);
		goto fail;
	}

	isc_log_write(ns_g_lctx, NS_LOGCATEGORY_GENERAL,
		NS_LOGMODULE_SERVER, ISC_LOG_INFO,
		"initializing GeoIP %s (type %d) DB", name, edition);

	db = GeoIP_open_type(edition, method);
	if (db == NULL) {
		isc_log_write(ns_g_lctx, NS_LOGCATEGORY_GENERAL,
			NS_LOGMODULE_SERVER, ISC_LOG_ERROR,
			"failed to initialize GeoIP %s (type %d) DB%s",
			name, edition, fallback == 0
			 ? "geoip matches using this database will fail" : "");
		goto fail;
	}

	info = GeoIP_database_info(db);
	if (info != NULL) {
		isc_log_write(ns_g_lctx, NS_LOGCATEGORY_GENERAL,
			      NS_LOGMODULE_SERVER, ISC_LOG_INFO,
			      "%s", info);
		free(info);
	}

	*dbp = db;
	return;
 fail:
	if (fallback != 0)
		init_geoip_db(dbp, fallback, 0, method, name);

}
#endif /* HAVE_GEOIP */

void
ns_geoip_init(void) {
#ifndef HAVE_GEOIP
	return;
#else
	GeoIP_cleanup();
	if (ns_g_geoip == NULL)
		ns_g_geoip = &geoip_table;
#endif
}

void
ns_geoip_load(char *dir) {
#ifndef HAVE_GEOIP

	UNUSED(dir);

	return;
#else
	GeoIPOptions method;

#ifdef _WIN32
	method = GEOIP_STANDARD;
#else
	method = GEOIP_MMAP_CACHE;
#endif

	ns_geoip_init();
	if (dir != NULL) {
		isc_log_write(ns_g_lctx, NS_LOGCATEGORY_GENERAL,
			      NS_LOGMODULE_SERVER, ISC_LOG_INFO,
			      "using \"%s\" as GeoIP directory", dir);
		GeoIP_setup_custom_directory(dir);
	}

	init_geoip_db(&ns_g_geoip->country_v4, GEOIP_COUNTRY_EDITION, 0,
		      method, "Country (IPv4)");
#ifdef HAVE_GEOIP_V6
	init_geoip_db(&ns_g_geoip->country_v6, GEOIP_COUNTRY_EDITION_V6, 0,
		      method, "Country (IPv6)");
#endif

	init_geoip_db(&ns_g_geoip->city_v4, GEOIP_CITY_EDITION_REV1,
		      GEOIP_CITY_EDITION_REV0, method, "City (IPv4)");
#if defined(HAVE_GEOIP_V6) && defined(HAVE_GEOIP_CITY_V6)
	init_geoip_db(&ns_g_geoip->city_v6, GEOIP_CITY_EDITION_REV1_V6,
		      GEOIP_CITY_EDITION_REV0_V6, method, "City (IPv6)");
#endif

	init_geoip_db(&ns_g_geoip->region, GEOIP_REGION_EDITION_REV1,
		      GEOIP_REGION_EDITION_REV0, method, "Region");

	init_geoip_db(&ns_g_geoip->isp, GEOIP_ISP_EDITION, 0,
		      method, "ISP");
	init_geoip_db(&ns_g_geoip->org, GEOIP_ORG_EDITION, 0,
		      method, "Org");
	init_geoip_db(&ns_g_geoip->as, GEOIP_ASNUM_EDITION, 0,
		      method, "AS");
	init_geoip_db(&ns_g_geoip->domain, GEOIP_DOMAIN_EDITION, 0,
		      method, "Domain");
	init_geoip_db(&ns_g_geoip->netspeed, GEOIP_NETSPEED_EDITION, 0,
		      method, "NetSpeed");
#endif /* HAVE_GEOIP */
}
