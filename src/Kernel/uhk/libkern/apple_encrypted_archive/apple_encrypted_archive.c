/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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
#include <libkern/libkern.h>
#include <libkern/section_keywords.h>
#include <libkern/apple_encrypted_archive/apple_encrypted_archive.h>

#if defined(SECURITY_READ_ONLY_LATE)
SECURITY_READ_ONLY_LATE(const apple_encrypted_archive_t *) apple_encrypted_archive = NULL;
SECURITY_READ_ONLY_LATE(registration_callback_t) registration_callback = NULL;
#else
const apple_encrypted_archive_t *apple_encrypted_archive = NULL;
registration_callback_t registration_callback = NULL;
#endif

void
apple_encrypted_archive_interface_register(const apple_encrypted_archive_t *aea)
{
	if (apple_encrypted_archive) {
		panic("apple_encrypted_archive interface already set");
	}

	apple_encrypted_archive = aea;

	if (registration_callback) {
		registration_callback();
	}
}

void
apple_encrypted_archive_interface_set_registration_callback(registration_callback_t callback)
{
	if (callback && registration_callback) {
		panic("apple_encrypted_archive interface registration callback is already set");
	}

	registration_callback = callback;
}
