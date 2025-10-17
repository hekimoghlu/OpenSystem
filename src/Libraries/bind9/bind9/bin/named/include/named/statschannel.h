/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
/* $Id: statschannel.h,v 1.3 2008/04/03 05:55:51 marka Exp $ */

#ifndef NAMED_STATSCHANNEL_H
#define NAMED_STATSCHANNEL_H 1

/*! \file
 * \brief
 * The statistics channels built-in the name server.
 */

#include <isccc/types.h>

#include <isccfg/aclconf.h>

#include <named/types.h>

#define NS_STATSCHANNEL_HTTPPORT		80

isc_result_t
ns_statschannels_configure(ns_server_t *server, const cfg_obj_t *config,
			   cfg_aclconfctx_t *aclconfctx);
/*%<
 * [Re]configure the statistics channels.
 *
 * If it is no longer there but was previously configured, destroy
 * it here.
 *
 * If the IP address or port has changed, destroy the old server
 * and create a new one.
 */


void
ns_statschannels_shutdown(ns_server_t *server);
/*%<
 * Initiate shutdown of all the statistics channel listeners.
 */

isc_result_t
ns_stats_dump(ns_server_t *server, FILE *fp);
/*%<
 * Dump statistics counters managed by the server to the file fp.
 */

#endif	/* NAMED_STATSCHANNEL_H */
