/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
**
**  NAME:
**
**      wc16str.c
**
**  FACILITY:
**
**      Microsoft RPC compatibility wrappers.
**
**  ABSTRACT:
**
**  This module converts between UTF8 and UTF16 encodings.
**
*/

#include "compat/mswrappers.h"

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <CoreFoundation/CFStringEncodingConverter.h>
#include <stdlib.h>
#include <string.h>

#include "wc16str.h"

#define CONVERSION_FLAGS ( \
    kCFStringEncodingAllowLossyConversion \
)

static size_t
wchar16_strlen(const wchar16_t *source)
{
    size_t len = 0;

    while (*source++) {
        ++len;
    }

    return len;
}

/* Convert from UTF16 (native endian) to UTF16. */
char *
awc16stombs(const wchar16_t * input)
{
    CFIndex inputlen = wchar16_strlen(input);
    char * output;

    uint32_t ret;
    CFIndex produced = 0;   // output units produced

    output = malloc(3 * (inputlen + 1));
    if (output == NULL) {
	return NULL;
    }

    ret = CFStringEncodingUnicodeToBytes(
	    kCFStringEncodingUTF8,
	    CONVERSION_FLAGS,
	    input, inputlen,
	    &inputlen,
	    (uint8_t *)output, inputlen * 3,
	    &produced);

    if (ret != kCFStringEncodingConversionSuccess) {
	free(output);
	return NULL;
    }

    output[produced] = '\0';
    return output;
}

wchar16_t *
ambstowc16s(const char * input)
{
    CFIndex inputlen = strlen(input);
    wchar16_t * output;

    uint32_t ret;
    CFIndex produced = 0;   // output units produced

    output = malloc(sizeof(wchar16_t) *(inputlen + 1));
    if (output == NULL) {
	return NULL;
    }

    ret = CFStringEncodingBytesToUnicode(
	    kCFStringEncodingUTF8,
	    CONVERSION_FLAGS,
	    (const uint8_t *)input, inputlen,
	    &inputlen,
	    output, inputlen * sizeof(wchar16_t),
	    &produced);

    if (ret != kCFStringEncodingConversionSuccess) {
	free(output);
	return NULL;
    }

    output[produced] = '\0';
    return output;
}
