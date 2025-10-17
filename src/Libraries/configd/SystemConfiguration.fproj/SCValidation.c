/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
 * Modification History
 *
 * March 14, 2025		Dieter Siegmund <dieter@apple.com>
 * - extracted from SCDHostName.c to make it easier to test
 */
#include <SCPrivate.h>
#include <SCValidation.h>

#define _DNS_MAX_LABEL_LEN	63
#define _DNS_MAX_LABELS		127
#define _DNS_MAX_LEN		255

Boolean
_SC_stringIsValidDNSName(const char *name)
{
	size_t		i;
	size_t		label_len = 0;
	size_t		len;
	size_t		ndots = 0;
	char		prev = '\0';
	const char	*scan;
	Boolean		valid = FALSE;

	len = (name == NULL) ? 0 : strlen(name);	
	if (len == 0 || len > _DNS_MAX_LEN) {
		goto done;
	}

	for (scan = name, i = 0; i < len; i++, scan++) {
		char	ch	= *scan;
		char 	next	= *(scan + 1);

		if (ch == '.') {
			ndots++;
			if (ndots > _DNS_MAX_LABELS) {
				/* too many labels */
				goto done;
			}
		} else {
			label_len++;
		}
		if ((prev == '.' || prev == '\0') && isalnum(ch) == 0) {
			/* a label must begin with a letter or digit */
			goto done;
		}
		if (next == '\0' || next == '.') {
			if (label_len > _DNS_MAX_LABEL_LEN) {
				goto done;
			}
			label_len = 0;
			if (ch == '.' && next == '\0' && ndots > 1) {
				/* trailing dot is OK e.g. "apple.com." */
				break;
			}
			if (isalnum(ch) == 0) {
				/* a label must end with a letter or digit */
				goto done;
			}
			if (next == '\0' && ndots == _DNS_MAX_LABELS) {
				/* too many labels */
				goto done;
			}
		} else if (isalnum(ch) == 0) {
			switch (ch) {
				case '.':
					/* a label separator */
					break;
				case '-':
					/* hyphens are OK within a label */
					break;
				default:
					/* an invalid character */
					goto done;
			}
		}
		prev = ch;
	}
	valid = TRUE;

 done:
	return valid;
}


Boolean
_SC_CFStringIsValidDNSName(CFStringRef name)
{
	Boolean	clean	= FALSE;
	char	*str	= NULL;

	if (!isA_CFString(name)) {
		return FALSE;
	}

	str = _SC_cfstring_to_cstring(name, NULL, 0, kCFStringEncodingASCII);
	if (str == NULL) {
		return FALSE;
	}

	clean = _SC_stringIsValidDNSName(str);

	if (str != NULL)	CFAllocatorDeallocate(NULL, str);
	return clean;
}



Boolean
_SC_CFStringIsValidNetBIOSName(CFStringRef name)
{
	if (!isA_CFString(name)) {
		return FALSE;
	}

	if (CFStringGetLength(name) > 15) {
		return FALSE;
	}

	return TRUE;
}


#ifdef TEST_VALID_DNS_NAMES

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define _10_CHARS	"abcdefghij"
#define _30_CHARS	_10_CHARS _10_CHARS _10_CHARS
#define _60_CHARS	_30_CHARS _30_CHARS
#define _63_CHARS	_60_CHARS "123"

#define _10_LABELS	"a.b.c.d.e.f.g.h.i.j."
#define _50_LABELS	_10_LABELS _10_LABELS _10_LABELS _10_LABELS _10_LABELS
#define _100_LABELS	_50_LABELS _50_LABELS
#define _127_LABELS	_100_LABELS _10_LABELS _10_LABELS "t.u.v.w.x.y.z"

static const char * good_names[] = {
	"www.apple.com",
	"www.apple.com.",
	"apple.com",
	"apple.com.",
	"apple",
	"a-a",
	"1",
	_63_CHARS "." _63_CHARS "." _63_CHARS "." _63_CHARS,
	_127_LABELS,
	_127_LABELS "."
};
static size_t good_names_count = sizeof(good_names) / sizeof(good_names[0]);

static const char * bad_names[] = {
	NULL,
	"",
	".",
	"..",
	"...",
	".com",
	".com.",
	"com.",
	".apple.com.",
	"-",
	"-.-",
	"a-",
	_63_CHARS "x",
	_127_LABELS ".x",
	_127_LABELS ".x.",
};
static size_t bad_names_count = sizeof(bad_names) / sizeof(bad_names[0]);


static void
iterate_names(const char * list[], size_t list_count, Boolean good)
{
	for (size_t i = 0; i < list_count; i++) {
		const char *	name = list[i];
		Boolean		valid;

		valid = _SC_stringIsValidDNSName(name);
		if (valid != good) {
			fprintf(stderr, "'%s' is supposed to be %s\n",
				name, good ? "good" : "bad");
			exit(2);
		}
	}
	printf("%zu %s tests PASS\n", list_count,
	       good ? "Good" : "Bad");
}

int
main(int argc, char * argv[])
{
	iterate_names(good_names, good_names_count, TRUE);
	iterate_names(bad_names, bad_names_count, FALSE);
}

#endif /* TEST_VALID_DNS_NAMES */

