/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
/* $Id: clientinfo.c,v 1.3 2011/10/11 00:25:12 marka Exp $ */

/*! \file */

#include "config.h"

#include <dns/clientinfo.h>

void
dns_clientinfomethods_init(dns_clientinfomethods_t *methods,
			   dns_clientinfo_sourceip_t sourceip)
{
	methods->version = DNS_CLIENTINFOMETHODS_VERSION;
	methods->age = DNS_CLIENTINFOMETHODS_AGE;
	methods->sourceip = sourceip;
}

void
dns_clientinfo_init(dns_clientinfo_t *ci, void *data) {
	ci->version = DNS_CLIENTINFO_VERSION;
	ci->data = data;
}
