/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
/*
 * SHA-1 in C
 * By Steve Reid <steve@edmweb.com>
 * 100% Public Domain
 */

#ifndef _SYS_SHA1_H_
#define	_SYS_SHA1_H_

typedef unsigned int  my_int32_t;
typedef unsigned char my_char;

typedef struct {
	my_int32_t state[5];
	my_int32_t count[2];  
	my_char    buffer[64];
} SHA1_CTX;
  
void	SHA1Transform(my_int32_t state[5], const my_char buffer[64]);
void	SHA1Init(SHA1_CTX *context);
void	SHA1Update(SHA1_CTX *context, const my_char *data, my_int32_t len);
void	SHA1Final(my_char digest[20], SHA1_CTX *context);

#endif /* _SYS_SHA1_H_ */
