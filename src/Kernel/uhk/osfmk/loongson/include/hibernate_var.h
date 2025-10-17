/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
 * Copyright (c) 2013 Paul Irofti.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

/* Loongson hibernate support definitions */

#define PAGE_MASK_4M ((256 * PAGE_SIZE) - 1)

#define PIGLET_PAGE_MASK ~((paddr_t)PAGE_MASK_4M)

/*
 * Steal hibernate pages right after the first page which is reserved
 * for the exception area.
 * */
#define HIBERNATE_STACK_PAGE	(PAGE_SIZE * 1)
#define HIBERNATE_INFLATE_PAGE	(PAGE_SIZE * 2)
#define HIBERNATE_COPY_PAGE	(PAGE_SIZE * 3)
#define HIBERNATE_HIBALLOC_PAGE	(PAGE_SIZE * 4)

#define HIBERNATE_RESERVED_PAGES	4

/* Use 4MB hibernation chunks */
#define HIBERNATE_CHUNK_SIZE		0x400000

#define HIBERNATE_CHUNK_TABLE_SIZE	0x100000
