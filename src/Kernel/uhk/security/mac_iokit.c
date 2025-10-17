/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
/*-
 * Copyright (c) 2006 SPARTA, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <sys/param.h>
#include <sys/types.h>
#include <sys/vnode.h>
#include <sys/vnode_internal.h>
#include <sys/queue.h>
#include <bsd/bsm/audit.h>
#include <bsd/security/audit/audit.h>
#include <bsd/sys/malloc.h>
#include <vm/vm_kern.h>

#include <security/mac_framework.h>
#include <security/mac_internal.h>

int
mac_iokit_check_open_service(kauth_cred_t cred, io_object_t service, unsigned int user_client_type)
{
	int error;

	MAC_CHECK(iokit_check_open_service, cred, service, user_client_type);
	return error;
}

int
mac_iokit_check_open(kauth_cred_t cred, io_object_t user_client, unsigned int user_client_type)
{
	int error;

	MAC_CHECK(iokit_check_open, cred, user_client, user_client_type);
	return error;
}

int
mac_iokit_check_set_properties(kauth_cred_t cred, io_object_t registry_entry, io_object_t properties)
{
	int error;

	MAC_CHECK(iokit_check_set_properties, cred, registry_entry, properties);
	return error;
}

int
mac_iokit_check_filter_properties(kauth_cred_t cred, io_object_t registry_entry)
{
	int error;

	MAC_CHECK(iokit_check_filter_properties, cred, registry_entry);
	return error;
}

int
mac_iokit_check_get_property(kauth_cred_t cred, io_object_t registry_entry, const char *name)
{
	int error;

	MAC_CHECK(iokit_check_get_property, cred, registry_entry, name);
	return error;
}

int
mac_iokit_check_hid_control(kauth_cred_t cred)
{
	int error;

	MAC_CHECK(iokit_check_hid_control, cred);
	return error;
}
