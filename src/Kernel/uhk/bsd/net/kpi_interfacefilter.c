/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#include "kpi_interfacefilter.h"

#include <sys/malloc.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <sys/kern_event.h>
#include <net/dlil.h>

#undef iflt_attach
errno_t
iflt_attach(
	ifnet_t interface,
	const struct iff_filter *filter,
	interface_filter_t *filter_ref);


errno_t
iflt_attach_internal(
	ifnet_t interface,
	const struct iff_filter *filter,
	interface_filter_t *filter_ref)
{
	if (interface == NULL) {
		return ENOENT;
	}

	return dlil_attach_filter(interface, filter, filter_ref,
	           DLIL_IFF_INTERNAL);
}

errno_t
iflt_attach(
	ifnet_t interface,
	const struct iff_filter *filter,
	interface_filter_t *filter_ref)
{
	if (interface == NULL) {
		return ENOENT;
	}

	return dlil_attach_filter(interface, filter, filter_ref, 0);
}

void
iflt_detach(
	interface_filter_t filter_ref)
{
	dlil_detach_filter(filter_ref);
}
