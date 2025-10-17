/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include <libkern/compression/compression.h>

#if defined(SECURITY_READ_ONLY_LATE)
SECURITY_READ_ONLY_LATE(const compression_ki_t*) compression_ki_ptr = NULL;
static SECURITY_READ_ONLY_LATE(registration_callback_t) registration_callback = NULL;
#else
const compression_ki_t* compression_ki_ptr = NULL;
static registration_callback_t registration_callback = NULL;
#endif

void
compression_interface_register(const compression_ki_t *ki)
{
	if (compression_ki_ptr) {
		panic("compression interface already set");
	}

	compression_ki_ptr = ki;

	if (registration_callback) {
		registration_callback();
	}
}

void
compression_interface_set_registration_callback(registration_callback_t callback)
{
	if (callback && registration_callback) {
		panic("compression interface registration callback is already set");
	}

	registration_callback = callback;
}
