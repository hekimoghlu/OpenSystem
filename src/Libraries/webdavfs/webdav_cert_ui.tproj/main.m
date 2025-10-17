/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <Security/oidscert.h>
#include <Security/cssmapi.h>
#include <Security/cssmapple.h>
#include <Security/certextensions.h>
#include <CoreFoundation/CFPropertyList.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFArray.h>
#include <SystemConfiguration/SCValidation.h>
#include <string.h>
#include <AssertMacros.h>

static void *
read_fd(int fd, size_t * ret_size)
{
	void *buf;
	size_t offset;
	size_t remaining;
	size_t size;

	size = 4096;
	buf = malloc(size);
	require(buf != NULL, malloc);

	offset = 0;
	remaining = size - offset;
	while ( TRUE )
	{
		size_t read_count;

		if (remaining == 0)
		{
			size *= 2;
			buf = reallocf(buf, size);
			require(buf != NULL, reallocf);
			
			remaining = size - offset;
		}
		
		read_count = read(fd, buf + offset, remaining);
		require(read_count >= 0, read);
		
		if (read_count == 0)
		{
			/* EOF */
			break;
		}
		
		offset += read_count;
		remaining -= read_count;
	}
	
	require(offset != 0, no_input);
	
	*ret_size = offset;
	return ( buf );


	/* error cases handled here */

no_input:
read:

	free(buf);

reallocf:	
malloc:

	*ret_size = 0;
	return ( NULL );
}

CFPropertyListRef 
my_CFPropertyListCreateFromFileDescriptor(int fd)
{
	void *buf;
	size_t bufsize;
	CFDataRef data;
	CFPropertyListRef plist;

	plist = NULL;

	buf = read_fd(fd, &bufsize);
	require(buf != NULL, read_fd);

	data = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, buf, bufsize, kCFAllocatorNull);
	require(data != NULL, CFDataCreateWithBytesNoCopy);

	plist = CFPropertyListCreateFromXMLData(kCFAllocatorDefault, data, kCFPropertyListImmutable, NULL);

	CFRelease(data);

CFDataCreateWithBytesNoCopy:

	free(buf);

read_fd:

	return (plist);
}

CFDictionaryRef the_dict;

/*
 * This program exits with one of the following values:
 * 0 = Continue button selected
 * 1 = Cancel button selected
 * 2 = an unexpected error was encountered
 */
int main(int argc, char *argv[])
{
	the_dict = my_CFPropertyListCreateFromFileDescriptor(STDIN_FILENO);
	if (isA_CFDictionary(the_dict) == NULL)
	{
		return ( 2 );
	}
	else
	{
		return ( NSApplicationMain(argc, (const char **) argv) );
	}
}
