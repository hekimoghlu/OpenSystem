/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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
#ifndef _KEXT_PANIC_REPORT_H_
#define _KEXT_PANIC_REPORT_H_

#include <sys/cdefs.h>

__BEGIN_DECLS

/*******************************************************************************
* String-compaction tables for panic reports' kext listing.
*******************************************************************************/

typedef struct subs_entry_t {
	const char * substring;
	char         substitute;
} subs_entry_t;

/* Prefix substitution list. Common prefixes are replaced with a single
 * nonalphanumeric character at the beginning of the identifier.
 *
 * List should be in descending order of # components, and should then
 * be in descending frequency order.
 */
subs_entry_t kext_identifier_prefix_subs[] = {
	{ "com.apple.driver.", '>' },
	{ "com.apple.iokit.", '|' },
	{ "com.apple.security.", '$' },
	{ "com.apple.", '@' },

	{ (char *)NULL, '\0' }
};

/* Substring substitution list. Substrings are replaced with a '!' followed
 * by a single letter mapping to the original string.
 *
 * List should be in descending frequency order, and within
 * groups containing same prefix, in descending length order.
 */
subs_entry_t kext_identifier_substring_subs[] = {
	{ "AppleUSB", 'U' },
	{ "Apple", 'A' },
	{ "Family", 'F' },
	{ "Storage", 'S' },
	{ "Controller", 'C' },
	{ "Bluetooth", 'B' },
	{ "Intel", 'I' },

	{ (char *)NULL, '\0' }
};

__END_DECLS
#endif /* _KEXT_PANIC_REPORT_H_ */
