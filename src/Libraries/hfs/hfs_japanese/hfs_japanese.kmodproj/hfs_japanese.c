/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#include <mach/mach_types.h>
#include <mach/kmod.h>

#include "CFStub.h"
#include <hfs/hfs_encodings.h>


int
MacJapaneseToUnicode(Str31 hfs_str, UniChar *uni_str, UniCharCount maxCharLen, UniCharCount *usedCharLen)
{
	UInt32 processedChars;

	processedChars = __CFFromMacJapanese(kCFStringEncodingUseCanonical | kCFStringEncodingUseHFSPlusCanonical,
					&hfs_str[1],
					hfs_str[0],
					uni_str,
					maxCharLen,
					usedCharLen);

	if (processedChars == (UInt32)hfs_str[0])
		return (0);
	else
		return (-1);
}

int
UnicodeToMacJapanese(UniChar *uni_str, UniCharCount unicodeChars, Str31 hfs_str)
{
	UniCharCount srcCharsUsed;
	UInt32 usedByteLen = 0;

        srcCharsUsed = __CFToMacJapanese(kCFStringEncodingComposeCombinings | kCFStringEncodingUseHFSPlusCanonical,
					uni_str,
					unicodeChars,
					(UInt8*)&hfs_str[1],
					sizeof(Str31) - 1,
					&usedByteLen);

	hfs_str[0] = usedByteLen;

	if (srcCharsUsed == unicodeChars)
		return (0);
	else
		return (-1);
}


__private_extern__ int
hfs_japanese_start(kmod_info_t *ki, void *data)
{
	int result;

	result = hfs_addconverter(ki->id, kCFStringEncodingMacJapanese,
			MacJapaneseToUnicode, UnicodeToMacJapanese);

	return (result == 0 ? KERN_SUCCESS : KERN_FAILURE);
}

__private_extern__ int
hfs_japanese_stop(kmod_info_t *ki, void *data)
{
	int result;

	result = hfs_remconverter(ki->id, kCFStringEncodingMacJapanese);

	return (result == 0 ? KERN_SUCCESS : KERN_FAILURE);
}



