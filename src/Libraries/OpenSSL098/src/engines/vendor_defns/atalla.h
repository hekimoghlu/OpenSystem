/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
typedef struct ItemStr
	{
	unsigned char *data;
	int len;
	} Item;

typedef struct RSAPrivateKeyStr
	{
	void *reserved;
	Item version;
	Item modulus;
	Item publicExponent;
	Item privateExponent;
	Item prime[2];
	Item exponent[2];
	Item coefficient;
	} RSAPrivateKey;

/* Predeclare the function pointer types that we dynamically load from the DSO.
 * These use the same names and form that Ben's original support code had (in
 * crypto/bn/bn_exp.c) unless of course I've inadvertently changed the style
 * somewhere along the way!
 */

typedef int tfnASI_GetPerformanceStatistics(int reset_flag,
					unsigned int *ret_buf);

typedef int tfnASI_GetHardwareConfig(long card_num, unsigned int *ret_buf);

typedef int tfnASI_RSAPrivateKeyOpFn(RSAPrivateKey * rsaKey,
					unsigned char *output,
					unsigned char *input,
					unsigned int modulus_len);

