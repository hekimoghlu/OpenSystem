/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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
/* $Id: config.h,v 1.16 2009/06/11 23:47:55 tbox Exp $ */

#ifndef NAMED_CONFIG_H
#define NAMED_CONFIG_H 1

/*! \file */

#include <isccfg/cfg.h>

#include <dns/types.h>
#include <dns/zone.h>

isc_result_t
ns_config_parsedefaults(cfg_parser_t *parser, cfg_obj_t **conf);

isc_result_t
ns_config_get(cfg_obj_t const * const *maps, const char *name,
	      const cfg_obj_t **obj);

isc_result_t
ns_checknames_get(const cfg_obj_t **maps, const char *name,
		  const cfg_obj_t **obj);

int
ns_config_listcount(const cfg_obj_t *list);

isc_result_t
ns_config_getclass(const cfg_obj_t *classobj, dns_rdataclass_t defclass,
		   dns_rdataclass_t *classp);

isc_result_t
ns_config_gettype(const cfg_obj_t *typeobj, dns_rdatatype_t deftype,
		  dns_rdatatype_t *typep);

dns_zonetype_t
ns_config_getzonetype(const cfg_obj_t *zonetypeobj);

isc_result_t
ns_config_getiplist(const cfg_obj_t *config, const cfg_obj_t *list,
		    in_port_t defport, isc_mem_t *mctx,
		    isc_sockaddr_t **addrsp, isc_dscp_t **dscpsp,
		    isc_uint32_t *countp);

void
ns_config_putiplist(isc_mem_t *mctx, isc_sockaddr_t **addrsp,
		    isc_dscp_t **dscpsp, isc_uint32_t count);

isc_result_t
ns_config_getipandkeylist(const cfg_obj_t *config, const cfg_obj_t *list,
			  isc_mem_t *mctx, isc_sockaddr_t **addrsp,
			  isc_dscp_t **dscpp, dns_name_t ***keys,
			  isc_uint32_t *countp);

void
ns_config_putipandkeylist(isc_mem_t *mctx, isc_sockaddr_t **addrsp,
			  isc_dscp_t **dscpsp, dns_name_t ***keys,
			  isc_uint32_t count);

isc_result_t
ns_config_getport(const cfg_obj_t *config, in_port_t *portp);

isc_result_t
ns_config_getkeyalgorithm(const char *str, dns_name_t **name,
			  isc_uint16_t *digestbits);
isc_result_t
ns_config_getkeyalgorithm2(const char *str, dns_name_t **name,
			   unsigned int *typep, isc_uint16_t *digestbits);

isc_result_t
ns_config_getdscp(const cfg_obj_t *config, isc_dscp_t *dscpp);

#endif /* NAMED_CONFIG_H */
