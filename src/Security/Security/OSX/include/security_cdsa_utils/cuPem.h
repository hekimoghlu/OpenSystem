/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
	File:		 cuPem.h 
	
	Description: PEM encode/decode routines

	Author:		 dmitch
*/

#ifdef __cplusplus
extern "C" {
#endif

int isPem(
	const unsigned char 	*inData,
	unsigned 				inDataLen);

int pemEncode(
	const unsigned char 	*inData,
	unsigned 				inFileLen,
	unsigned char 			**outData,
	unsigned 				*outDataLen,
	const char 				*headerString);

int pemDecode(
	const unsigned char 	*inData,
	unsigned 				inFileLen,
	unsigned char 			**outData,
	unsigned 				*outDataLen);

#ifdef __cplusplus
}
#endif
