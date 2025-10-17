/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include "scutil.h"
#include "dictionary.h"


__private_extern__
void
do_dictInit(int argc, char * const argv[])
{
#pragma unused(argc)
#pragma unused(argv)
	if (value != NULL) {
		CFRelease(value);
	}

	value = CFDictionaryCreateMutable(NULL
					 ,0
					 ,&kCFTypeDictionaryKeyCallBacks
					 ,&kCFTypeDictionaryValueCallBacks
					 );

	return;
}


__private_extern__
void
do_dictShow(int argc, char * const argv[])
{
#pragma unused(argc)
#pragma unused(argv)
	if (value == NULL) {
		SCPrint(TRUE, stdout, CFSTR("d.show: dictionary must be initialized.\n"));
		return;
	}

	SCPrint(TRUE, stdout, CFSTR("%@\n"), value);

	return;
}


__private_extern__
void
do_dictSetKey(int argc, char * const argv[])
{
	CFMutableArrayRef	array		= NULL;
	Boolean			doArray		= FALSE;
	Boolean			doBoolean	= FALSE;
	Boolean			doData		= FALSE;
	Boolean			doNumeric	= FALSE;
	CFStringRef		key;
	CFMutableDictionaryRef	newValue;
	CFTypeRef		val		= NULL;

	if (value == NULL) {
		SCPrint(TRUE, stdout, CFSTR("d.add: dictionary must be initialized.\n"));
		return;
	}

	if (!isA_CFDictionary(value)) {
		SCPrint(TRUE, stdout, CFSTR("d.add: data (fetched from configuration server) is not a dictionary.\n"));
		return;
	}

	key = CFStringCreateWithCString(NULL, argv[0], kCFStringEncodingUTF8);
	argv++; argc--;

	while (argc > 0) {
		if (strcmp(argv[0], "*") == 0) {
			/* if array requested */
			doArray = TRUE;
		} else if (strcmp(argv[0], "-") == 0) {
			/* if string values follow */
			argv++; argc--;
			break;
		} else if (strcmp(argv[0], "?") == 0) {
			/* if boolean values requested */
			doBoolean = TRUE;
		} else if (strcmp(argv[0], "%") == 0) {
			/* if [hex] data values requested */
			doData = TRUE;
		} else if (strcmp(argv[0], "#") == 0) {
			/* if numeric values requested */
			doNumeric = TRUE;
		} else {
			/* it's not a special flag */
			break;
		}
		argv++; argc--;
	}

	if (argc > 1) {
		doArray = TRUE;
	} else if (!doArray && (argc == 0)) {
		SCPrint(TRUE, stdout, CFSTR("d.add: no values.\n"));
		CFRelease(key);
		return;
	}

	if (doArray) {
		array = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
	}

	while (argc > 0) {
		if (doBoolean) {
			if         ((strcasecmp(argv[0], "true") == 0) ||
				    (strcasecmp(argv[0], "t"   ) == 0) ||
				    (strcasecmp(argv[0], "yes" ) == 0) ||
				    (strcasecmp(argv[0], "y"   ) == 0) ||
				    (strcmp    (argv[0], "1"   ) == 0)) {
				val = CFRetain(kCFBooleanTrue);
			} else if ((strcasecmp(argv[0], "false") == 0) ||
				   (strcasecmp(argv[0], "f"    ) == 0) ||
				   (strcasecmp(argv[0], "no"   ) == 0) ||
				   (strcasecmp(argv[0], "n"    ) == 0) ||
				   (strcmp    (argv[0], "0"    ) == 0)) {
				val = CFRetain(kCFBooleanFalse);
			} else {
				SCPrint(TRUE, stdout, CFSTR("d.add: invalid data.\n"));
				if (doArray) CFRelease(array);
				CFRelease(key);
				return;
			}
		} else if (doData) {
			uint8_t			*bytes;
			CFMutableDataRef	data;
			int			i;
			int			j;
			int			n;

			n = (int)strlen(argv[0]);
			if ((n % 2) == 1) {
				SCPrint(TRUE, stdout, CFSTR("d.add: not enough bytes.\n"));
				if (doArray) CFRelease(array);
				CFRelease(key);
				return;
			}

			data = CFDataCreateMutable(NULL, (n / 2));
			CFDataSetLength(data, (n / 2));

			bytes = (uint8_t *)CFDataGetBytePtr(data);
			for (i = 0, j = 0; i < n; i += 2, j++) {
				unsigned long	byte;
				char		*end;
				char		str[3]	= { 0 };

				str[0] = argv[0][i];
				str[1] = argv[0][i + 1];
				errno = 0;
				byte = strtoul(str, &end, 16);
				if ((*end != '\0') || (errno != 0)) {
					CFRelease(data);
					data = NULL;
					break;
				}

				bytes[j] = byte;
			}

			if (data == NULL) {
				SCPrint(TRUE, stdout, CFSTR("d.add: invalid data.\n"));
				if (doArray) CFRelease(array);
				CFRelease(key);
				return;
			}

			val = data;
		} else if (doNumeric) {
			int	intValue;

			if (sscanf(argv[0], "%d", &intValue) == 1) {
				val = CFNumberCreate(NULL, kCFNumberIntType, &intValue);
			} else {
				SCPrint(TRUE, stdout, CFSTR("d.add: invalid data.\n"));
				if (doArray) CFRelease(array);
				CFRelease(key);
				return;
			}
		} else {
			val = (CFPropertyListRef)CFStringCreateWithCString(NULL, argv[0], kCFStringEncodingUTF8);
		}

		if (doArray) {
			CFArrayAppendValue(array, val);
			CFRelease(val);
			argv++; argc--;
		} else {
			break;
		}
	}

	newValue = CFDictionaryCreateMutableCopy(NULL, 0, value);
	if (doArray) {
		CFDictionarySetValue(newValue, key, array);
		CFRelease(array);
	} else if (val != NULL) {
		CFDictionarySetValue(newValue, key, val);
		CFRelease(val);
	}
	CFRelease(key);
	CFRelease(value);
	value = newValue;

	return;
}


__private_extern__
void
do_dictRemoveKey(int argc, char * const argv[])
{
#pragma unused(argc)
	CFStringRef		key;
	CFMutableDictionaryRef	val;

	if (value == NULL) {
		SCPrint(TRUE, stdout, CFSTR("d.remove: dictionary must be initialized.\n"));
		return;
	}

	if (!isA_CFDictionary(value)) {
		SCPrint(TRUE, stdout, CFSTR("d.remove: data (fetched from configuration server) is not a dictionary.\n"));
		return;
	}

	val = CFDictionaryCreateMutableCopy(NULL, 0, value);
	CFRelease(value);
	value = val;

	key = CFStringCreateWithCString(NULL, argv[0], kCFStringEncodingUTF8);
	CFDictionaryRemoveValue((CFMutableDictionaryRef)value, key);
	CFRelease(key);

	return;
}
