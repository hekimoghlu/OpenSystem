/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
struct mdfour {
	uint32 A, B, C, D;
	uint32 totalN;          /* bit count, lower 32 bits */
	uint32 totalN2;         /* bit count, upper 32 bits */
};

void mdfour_begin(struct mdfour *md);
void mdfour_update(struct mdfour *md, unsigned char *in, uint32 n);
void mdfour_result(struct mdfour *md, unsigned char *out);
void mdfour(unsigned char *out, unsigned char *in, int n);
