/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
 * Copyright (c) 1997-2000 Apple Inc. All Rights Reserved
 */

#ifndef _HFS_ENCODINGS_H_
#define _HFS_ENCODINGS_H_

#if !TARGET_OS_IPHONE

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_UNSTABLE

#define CTL_HFS_NAMES { \
	{ 0, 0 }, \
	{ "encodingbias", CTLTYPE_INT }, \
}

__BEGIN_DECLS

/*
 * HFS Filename Encoding Converters Interface
 *
 * Private Interface for adding hfs filename
 * encoding converters. These are not needed
 * for HFS Plus volumes (since they already
 * have Unicode filenames).
 *
 * Used by HFS Encoding Converter Kernel Modules
 * to register their encoding conversion routines.
 */

typedef int (* hfs_to_unicode_func_t)(const uint8_t hfs_str[32], uint16_t *uni_str,
		u_int32_t maxCharLen, u_int32_t *usedCharLen);

typedef int (* unicode_to_hfs_func_t)(uint16_t *uni_str, u_int32_t unicodeChars,
        uint8_t hfs_str[32]);

int hfs_relconverter (u_int32_t encoding);
int hfs_getconverter(u_int32_t encoding, hfs_to_unicode_func_t *get_unicode,
					 unicode_to_hfs_func_t *get_hfsname);
int hfs_addconverter(int kmod_id, u_int32_t encoding,
					 hfs_to_unicode_func_t get_unicode,
					 unicode_to_hfs_func_t get_hfsname);
int hfs_remconverter(int kmod_id, u_int32_t encoding);

u_int32_t hfs_pickencoding(const u_int16_t *src, int len);
u_int32_t hfs_getencodingbias(void);
void hfs_setencodingbias(u_int32_t bias);
int mac_roman_to_utf8(const uint8_t hfs_str[32], uint32_t maxDstLen, uint32_t *actualDstLen,
					  unsigned char* dstStr);
int utf8_to_mac_roman(uint32_t srcLen, const unsigned char* srcStr, uint8_t dstStr[32]);
int mac_roman_to_unicode(const uint8_t hfs_str[32], uint16_t *uni_str, u_int32_t maxCharLen, u_int32_t *usedCharLen);
int unicode_to_mac_roman(uint16_t *uni_str, u_int32_t unicodeChars, uint8_t hfs_str[32]);

__END_DECLS

#endif /* __APPLE_API_UNSTABLE */

#endif // !TARGET_OS_IPHONE

#endif /* ! _HFS_ENCODINGS_H_ */
