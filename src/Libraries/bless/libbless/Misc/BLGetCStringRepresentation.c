/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
 *  BLGetCStringRepresentation.c
 *  bless
 *
 *  Created by Shantonu Sen on 5/30/06.
 *  Copyright 2006-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include <CoreFoundation/CoreFoundation.h>

#include "bless.h"
#include "bless_private.h"

/*
 * For the given CFTypeRef, get a C-string representation, backed
 * by thread-local storage
 */

#if !defined(NO_GETCSTRING) || !NO_GETCSTRING

static pthread_once_t	blcstr_once_control = PTHREAD_ONCE_INIT;
static pthread_key_t	blcstr_key = 0;

static void initkey(void);
static void releasestorage(void *addr);

struct stringer {
	size_t		size;
	char	*	string;
};

char *BLGetCStringDescription(CFTypeRef typeRef) {

	CFStringRef desc = NULL;
	int ret;
	struct stringer	*storage;
	CFIndex	strsize;
	
	if(typeRef == NULL)
		return NULL;
	
	ret = pthread_once(&blcstr_once_control, initkey);
	if(ret)
		return NULL;

	if(CFGetTypeID(typeRef) == CFStringGetTypeID()) {
		desc = CFRetain(typeRef);
	} else {
		desc = CFCopyDescription(typeRef);		
	}
	if(desc == NULL)
		return NULL;
	
	strsize = CFStringGetLength(desc);
	
	// assume encoding size of 3x as UTF-8
	strsize = 3*strsize + 1;
	
	storage = (struct stringer	*)pthread_getspecific(blcstr_key);
	if(storage == NULL) {
		storage = malloc(sizeof(*storage));
		storage->size = (size_t)strsize;
		storage->string = malloc(storage->size);

		ret = pthread_setspecific(blcstr_key, storage);
		if(ret) {
			CFRelease(desc);
			free(storage->string);
			free(storage);
			fprintf(stderr, "pthread_setspecific failed\n");
			return NULL;
		}
		
	} else if(storage->size < strsize) {
		// need more space
		storage->size = (size_t)strsize;
		free(storage->string);
		storage->string = malloc(storage->size);
	}
	
	if(!CFStringGetCString(desc, storage->string, (CFIndex)storage->size, kCFStringEncodingUTF8)) {
		CFRelease(desc);
		fprintf(stderr, "CFStringGetCString failed\n");		
		return NULL;
	}
	
	CFRelease(desc);
	
	return storage->string;
}

static void initkey(void)
{
	int ret;
	
	ret = pthread_key_create(&blcstr_key, releasestorage);
	if(ret)
		fprintf(stderr, "pthread_key_create failed\n");
//	printf("pthread_key_create: %lu\n", blcstr_key);
}

static void releasestorage(void *addr)
{
	// should be non-NULL
	struct stringer	*storage = (struct stringer	*)addr;

	free(storage->string);
	free(storage);
}

#else

char *BLGetCStringDescription(CFTypeRef typeRef) {
	return NULL;
}
#endif
