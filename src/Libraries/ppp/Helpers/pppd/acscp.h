/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#ifndef __ACSCP_H__
#define __ACSCP_H__

#include "pppd.h"

/*
 * Options.
 */
#define CI_ROUTES		1	/* Remote Routes */
#define	CI_DOMAINS		2	/* Remote DNS Domains */

#define	LATEST_ROUTES_VERSION	1
#define LATEST_DOMAINS_VERSION	1

typedef struct acscp_options {
    bool		neg_routes;
    u_int32_t		routes_version;		/* version for routing data format */
    bool		neg_domains;	
    u_int32_t		domains_version;	/* version for domains format */
} acscp_options;


extern acscp_options acscp_wantoptions[];
extern acscp_options acscp_gotoptions[];
extern acscp_options acscp_allowoptions[];
extern acscp_options acscp_hisoptions[];

extern fsm acscp_fsm[];
extern struct protent acscp_protent;

#endif
