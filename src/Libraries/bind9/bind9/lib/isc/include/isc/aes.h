/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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

/*! \file isc/aes.h */

#ifndef ISC_AES_H
#define ISC_AES_H 1

#include <isc/lang.h>
#include <isc/platform.h>
#include <isc/types.h>

#define ISC_AES128_KEYLENGTH 16U
#define ISC_AES192_KEYLENGTH 24U
#define ISC_AES256_KEYLENGTH 32U
#define ISC_AES_BLOCK_LENGTH 16U

#ifdef ISC_PLATFORM_WANTAES

ISC_LANG_BEGINDECLS

void
isc_aes128_crypt(const unsigned char *key, const unsigned char *in,
		 unsigned char *out);

void
isc_aes192_crypt(const unsigned char *key, const unsigned char *in,
		 unsigned char *out);

void
isc_aes256_crypt(const unsigned char *key, const unsigned char *in,
		 unsigned char *out);

ISC_LANG_ENDDECLS

#endif /* ISC_PLATFORM_WANTAES */

#endif /* ISC_AES_H */
