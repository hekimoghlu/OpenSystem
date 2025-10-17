/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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

#ifndef _rk_HEX_H_
#define _rk_HEX_H_ 1

#ifndef ROKEN_LIB_FUNCTION
#ifdef _WIN32
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL     __cdecl
#else
#define ROKEN_LIB_FUNCTION
#define ROKEN_LIB_CALL
#endif
#endif

#define hex_encode rk_hex_encode
#define hex_encode_lower rk_hex_encode_lower
#define hex_decode rk_hex_decode

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
	hex_encode(const void *, size_t, char **);
ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
	hex_encode_lower(const void *data, size_t size, char **str);
ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
	hex_decode(const char *, void *, size_t);

#endif /* _rk_HEX_H_ */
