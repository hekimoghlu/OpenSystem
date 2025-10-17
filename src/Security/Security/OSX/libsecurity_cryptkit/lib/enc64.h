/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#ifndef	_CK_ENC64_H_
#define _CK_ENC64_H_

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Given input buffer inbuf, length inlen, decode from 64-char IA5 format to
 * binary. Result is fmalloced and returned; its length is returned in *outlen.
 * NULL return indicates corrupted input.
 */
unsigned char *enc64(const unsigned char *inbuf,
	unsigned inlen,
	unsigned *outlen);		// RETURNED

/*
 * Enc64, with embedded newlines every lineLen in result. A newline is
 * the Microsoft-style "\r\n".
 */
unsigned char *enc64WithLines(const unsigned char *inbuf,
	unsigned inlen,
	unsigned linelen,
	unsigned *outlen);		// RETURNED

/*
 * Given input buffer inbuf, length inlen, decode from 64-char IA5 format to
 * binary. Result is fmalloced and returned; its length is returned in *outlen.
 * NULL return indicates corrupted input. All whitespace in inbuf is
 * ignored.
 */
unsigned char *dec64(const unsigned char *inbuf,
	unsigned inlen,
	unsigned *outlen);

/*
 * Determine if specified input data is valid enc64 format. Returns 1
 * if valid, 0 if not.
 */
int isValidEnc64(const unsigned char *inbuf,
	unsigned inbufLen);

#ifdef __cplusplus
}
#endif

#endif	/*_CK_ENC64_H_*/
