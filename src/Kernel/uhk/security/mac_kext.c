/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#include <sys/param.h>
#include <sys/kauth.h>
#include <security/mac_framework.h>
#include <security/mac_internal.h>

int
mac_kext_check_load(kauth_cred_t cred, const char *identifier)
{
	int error;

	MAC_CHECK(kext_check_load, cred, identifier);

	return error;
}

int
mac_kext_check_unload(kauth_cred_t cred, const char *identifier)
{
	int error;

	MAC_CHECK(kext_check_unload, cred, identifier);

	return error;
}

int
mac_kext_check_query(kauth_cred_t cred)
{
	int error;

	MAC_CHECK(kext_check_query, cred);

	return error;
}
