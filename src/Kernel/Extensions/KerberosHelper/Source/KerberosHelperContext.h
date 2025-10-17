/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
/*
 * Copyright (c) 2006-2007 Apple Inc. All rights reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 *
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 *
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 *
 * @APPLE_LICENSE_HEADER_END@
 */

#include <Heimdal/krb5.h>
#include <Heimdal/hx509.h>

struct realm_mappings {
	int lkdc;
	char *hostname;
	char *realm;
};


typedef struct KRBhelperContext {
	CFStringRef     inHostName; /* User input string, still printable */
	CFStringRef     hostname; /* calculated hostname */
	struct addrinfo *addr;
	CFStringRef	realm;  /* calculated realmname */
        struct {
	    struct realm_mappings *data;
	    size_t len;
	} realms;
	CFStringRef	inAdvertisedPrincipal;
	char            *useName, *useInstance, *useRealm, *defaultRealm;
	krb5_context	krb5_ctx;
	hx509_context	hx_ctx;
	unsigned	noGuessing:1;
} KRBhelperContext;

OSStatus
KRBCredChangeReferenceCount(CFStringRef clientPrincipal, int change, int excl);

#define kGSSAPIMechSupportsAppleLKDC	    CFSTR("1.2.752.43.14.3")
