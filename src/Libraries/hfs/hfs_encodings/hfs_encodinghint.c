/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#include <sys/param.h>
#include "../core/hfs.h"


/* CJK Mac Encoding Bits */
#define CJK_JAPAN	        0x1
#define CJK_KOREAN	        0x2
#define CJK_CHINESE_TRAD	0x4
#define CJK_CHINESE_SIMP	0x8
#define CJK_ALL	            0xF

#define CJK_CHINESE    (CJK_CHINESE_TRAD | CJK_CHINESE_SIMP)
#define CJK_KATAKANA   (CJK_JAPAN)


/* Remember the last unique CJK bit */
u_int8_t cjk_lastunique = 0;

/* Encoding bias */
u_int32_t hfs_encodingbias = 0;
int hfs_islatinbias = 0;

extern lck_mtx_t  encodinglst_mutex;


/* Map CJK bits to Mac encoding */
u_int8_t cjk_encoding[] = {
	/* 0000 */  kTextEncodingMacUnicode,
	/* 0001 */  kTextEncodingMacJapanese,
	/* 0010 */  kTextEncodingMacKorean,
	/* 0011 */  kTextEncodingMacJapanese,
	/* 0100 */  kTextEncodingMacChineseTrad,
	/* 0101 */  kTextEncodingMacJapanese,
	/* 0110 */  kTextEncodingMacKorean,
	/* 0111 */  kTextEncodingMacJapanese,
	/* 1000 */  kTextEncodingMacChineseSimp,
	/* 1001 */  kTextEncodingMacJapanese,
	/* 1010 */  kTextEncodingMacKorean,
	/* 1011 */  kTextEncodingMacJapanese,
	/* 1100 */  kTextEncodingMacChineseTrad,
	/* 1101 */  kTextEncodingMacJapanese,
	/* 1110 */  kTextEncodingMacKorean,
	/* 1111 */  kTextEncodingMacJapanese
};


u_int32_t
hfs_pickencoding(__unused const u_int16_t *src, __unused int len) {
	/* Just return kTextEncodingMacRoman if HFS standard is not supported. */
	return kTextEncodingMacRoman;
}


u_int32_t hfs_getencodingbias(void)
{
	return (hfs_encodingbias);
}


void hfs_setencodingbias(u_int32_t bias)
{
	hfs_encodingbias = bias;

	switch (bias) {
	case kTextEncodingMacRoman:
	case kTextEncodingMacCentralEurRoman:
	case kTextEncodingMacTurkish:
	case kTextEncodingMacCroatian:
	case kTextEncodingMacIcelandic:
	case kTextEncodingMacRomanian:
		hfs_islatinbias = 1;
		break;
	default:
		hfs_islatinbias = 0;
		break;					
	}
}
