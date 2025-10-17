/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
#include "misc.h"
#include "SecCFRelease.h"

// NOTE: the return may or allocate a fair bit more space then it needs.
// Use it for short lived conversions (or strdup the result).
char *utf8(CFStringRef s) {
	CFIndex sz = CFStringGetMaximumSizeForEncoding(CFStringGetLength(s), kCFStringEncodingUTF8) + 1;
	CFIndex used = 0;
	UInt8 *buf = (UInt8 *)malloc(sz);
	if (!buf) {
		return NULL;
	}
	CFStringGetBytes(s, CFRangeMake(0, CFStringGetLength(s)), kCFStringEncodingUTF8, '?', FALSE, buf, sz, &used);
	buf[used] = 0;
	
	return (char*)buf;
}

void CFfprintf(FILE *f, CFStringRef format, ...) {
	va_list ap;
	va_start(ap, format);
	
	CFStringRef str = CFStringCreateWithFormatAndArguments(NULL, NULL, format, ap);
	va_end(ap);
	
	CFIndex sz = CFStringGetMaximumSizeForEncoding(CFStringGetLength(str), kCFStringEncodingUTF8);
	sz += 1;
	CFIndex used = 0;
	unsigned char *buf;
	bool needs_free = false;
	if (sz < 1024) {
		buf = alloca(sz);
	} else {
		buf = malloc(sz);
		needs_free = true;
	}
	if (buf) {
		CFStringGetBytes(str, CFRangeMake(0, CFStringGetLength(str)), kCFStringEncodingUTF8, '?', FALSE, buf, sz, &used);
	} else {
		buf = (unsigned char *)"malloc failue during CFfprintf\n";
        needs_free = false;
	}
	
	fwrite(buf, 1, used, f);
	if (needs_free) {
		free(buf);
	}
	
	CFReleaseNull(str);
}

CFErrorRef fancy_error(CFStringRef domain, CFIndex code, CFStringRef description) {
	const void *v_ekey = kCFErrorDescriptionKey;
	const void *v_description = description;
	CFErrorRef err = CFErrorCreateWithUserInfoKeysAndValues(NULL, domain, code, &v_ekey, &v_description, 1);
    CFAutorelease(err);
	
	return err;
}

void CFSafeRelease(CFTypeRef object) {
	if (object) {
		CFReleaseNull(object);
	}
}

