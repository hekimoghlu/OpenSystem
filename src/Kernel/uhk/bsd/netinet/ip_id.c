/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
/*-
 * Copyright (c) 2008 Michael J. Silbersack.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <sys/kernel.h>
#include <netinet/ip_var.h>
#include <sys/random.h>
#include <dev/random/randomdev.h>

uint16_t
ip_randomid(uint64_t salt)
{
	uint16_t new_id;
	/*
	 * Mostly the salt value is heap or stack pointer value.
	 * To avoid any chance of leaking address apply macro that
	 * adds per-boot random number to it.
	 */
	uint64_t salt_tmp = VM_KERNEL_ADDRPERM(salt);

	/*
	 * When the passed salt values are addresses, it is possible some bits
	 * to not change at all, for example some higher order bits may not change
	 * and some lower order bits may be all 0s given particular alignment and object
	 * sizes.
	 * Therefore we compute a XOR'd 16bit value by considering all the bits
	 * of the salt to increase the likelihood of it being different.
	 */
	uint16_t new_id_salt = (uint16_t)(((salt_tmp >> 48) ^ (salt_tmp >> 32) ^
	    (salt_tmp >> 16) ^ salt_tmp) & 0xFF);

	/* Avoid returning IP ID value of 0 */
	do {
		read_random(&new_id, sizeof(new_id));
	} while (new_id == new_id_salt);

	return new_id ^ new_id_salt;
}
