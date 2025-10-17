/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
/* $Id: check-tool.h,v 1.18 2011/12/09 23:47:02 tbox Exp $ */

#ifndef CHECK_TOOL_H
#define CHECK_TOOL_H

/*! \file */

#include <isc/lang.h>
#include <isc/stdio.h>
#include <isc/types.h>

#include <dns/masterdump.h>
#include <dns/types.h>

ISC_LANG_BEGINDECLS

isc_result_t
setup_logging(isc_mem_t *mctx, FILE *errout, isc_log_t **logp);

isc_result_t
load_zone(isc_mem_t *mctx, const char *zonename, const char *filename,
	  dns_masterformat_t fileformat, const char *classname,
	  dns_ttl_t maxttl, dns_zone_t **zonep);

isc_result_t
dump_zone(const char *zonename, dns_zone_t *zone, const char *filename,
	  dns_masterformat_t fileformat, const dns_master_style_t *style,
	  const isc_uint32_t rawversion);

#ifdef _WIN32
void InitSockets(void);
void DestroySockets(void);
#endif

extern int debug;
extern const char *journal;
extern isc_boolean_t nomerge;
extern isc_boolean_t docheckmx;
extern isc_boolean_t docheckns;
extern isc_boolean_t dochecksrv;
extern unsigned int zone_options;
extern unsigned int zone_options2;

ISC_LANG_ENDDECLS

#endif
