/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
/* $Id$ */

#include <config.h>

#include <gssapi_rewrite.h>

#include <krb5-types.h>

#include <sys/types.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>

#include <gssapi_asn1.h>
#include <der.h>

#include <roken.h>

#include <gssapi.h>
#include <gssapi_mech.h>
#include <gssapi_krb5.h>
#include <gssapi_scram.h>
#include <gssapi_spnego.h>
#include <gssapi_ntlm.h>
#include <gssapi_netlogon.h>
#include <gssapi_spi.h>
#include <GSSPrivate.h>

#include <heimbase.h>

#include "mechqueue.h"

#include "context.h"
#include "cred.h"
#include "mech_switch.h"
#include "name.h"
#include "utils.h"
#include "compat.h"

#define _mg_buffer_zero(buffer) \
	do {					\
		if (buffer) {			\
			(buffer)->value = NULL;	\
			(buffer)->length = 0;	\
		 }				\
	} while(0)

#define _mg_oid_set_zero(oid_set) \
	do {						\
		if (oid_set) {				\
			(oid_set)->elements = NULL;	\
			(oid_set)->count = 0;		\
		 }					\
	} while(0)
