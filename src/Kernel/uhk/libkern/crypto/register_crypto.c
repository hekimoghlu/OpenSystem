/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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
#include <libkern/crypto/register_crypto.h>
#include <libkern/crypto/crypto_internal.h>
#include <libkern/section_keywords.h>

SECURITY_READ_ONLY_LATE(bool) crypto_init = false;
SECURITY_READ_ONLY_LATE(crypto_functions_t) g_crypto_funcs = NULL;

int
register_crypto_functions(const crypto_functions_t funcs)
{
	if (g_crypto_funcs) {
		return -1;
	}

	g_crypto_funcs = funcs;
	crypto_init = true;

	return 0;
}
