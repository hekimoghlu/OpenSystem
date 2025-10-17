/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 22, 2022.
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
#ifndef ARCHIVE_CRC32_H
#define ARCHIVE_CRC32_H

#ifndef __LIBARCHIVE_BUILD
#error This header is only to be used internally to libarchive.
#endif

/*
 * When zlib is unavailable, we should still be able to validate
 * uncompressed zip archives.  That requires us to be able to compute
 * the CRC32 check value.  This is a drop-in compatible replacement
 * for crc32() from zlib.  It's slower than the zlib implementation,
 * but still pretty fast: This runs about 300MB/s on my 3GHz P4
 * compared to about 800MB/s for the zlib implementation.
 */
static unsigned long
crc32(unsigned long crc, const void *_p, size_t len)
{
	unsigned long crc2, b, i;
	const unsigned char *p = _p;
	static volatile int crc_tbl_inited = 0;
	static unsigned long crc_tbl[256];

	if (!crc_tbl_inited) {
		for (b = 0; b < 256; ++b) {
			crc2 = b;
			for (i = 8; i > 0; --i) {
				if (crc2 & 1)
					crc2 = (crc2 >> 1) ^ 0xedb88320UL;
				else    
					crc2 = (crc2 >> 1);
			}
			crc_tbl[b] = crc2;
		}
		crc_tbl_inited = 1;
	}

	crc = crc ^ 0xffffffffUL;
	/* A use of this loop is about 20% - 30% faster than
	 * no use version in any optimization option of gcc.  */
	for (;len >= 8; len -= 8) {
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
	}
	while (len--)
		crc = crc_tbl[(crc ^ *p++) & 0xff] ^ (crc >> 8);
	return (crc ^ 0xffffffffUL);
}

#endif
