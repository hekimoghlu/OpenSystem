/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
#ifndef __DNSSD_OBJECT_H__
#define __DNSSD_OBJECT_H__

#include "dnssd_private.h"

//======================================================================================================================
// MARK: - Object Private Method Declarations

char *
dnssd_object_copy_description(dnssd_any_t object, bool debug, bool privacy);

void
dnssd_object_finalize(dnssd_any_t object);

#define DNSSD_OBJECT_ALLOC_DECLARE(NAME)		\
	dnssd_ ## NAME ## _t						\
	dnssd_object_ ## NAME ## _alloc(size_t size)

DNSSD_OBJECT_ALLOC_DECLARE(getaddrinfo);
DNSSD_OBJECT_ALLOC_DECLARE(getaddrinfo_result);
DNSSD_OBJECT_ALLOC_DECLARE(cname_array);

#endif	// __DNSSD_OBJECT_H__
