/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
 * Copyright (c) 2009 Mark Kettenis
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

struct md_header {
	uint32_t	transport_version;
	uint32_t	node_blk_sz;
	uint32_t	name_blk_sz;
	uint32_t	data_blk_sz;
};

struct md_element {
	uint8_t		tag;
	uint8_t		name_len;
	uint16_t	_reserved_field;
	uint32_t	name_offset;
	union {
		struct {
			uint32_t	data_len;
			uint32_t	data_offset;
		} y;
		uint64_t	val;
	} d;
};

#ifdef _KERNEL

extern caddr_t mdesc;
extern size_t mdesc_len;

extern caddr_t pri;
extern size_t pri_len;

void	 mdesc_init(void);
uint64_t mdesc_get_prop_val(int, const char *);
const char *mdesc_get_prop_str(int, const char *);
const char *mdesc_get_prop_data(int, const char *, size_t *);
int	mdesc_find(const char *, uint64_t);
int	mdesc_find_child(int, const char *, uint64_t);
int	mdesc_find_node(const char *);

#endif
