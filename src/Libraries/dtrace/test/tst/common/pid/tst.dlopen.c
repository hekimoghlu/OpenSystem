/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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

#include <unistd.h>
#include <dlfcn.h>

int waiting(volatile int *a)
{
	return (*a);
}

/*
 * Value taken from pcre.h
 */
#define PCRE_CONFIG_UTF8 0

int main(void)
{
	volatile int a = 0;
	
	while (waiting(&a) == 0)
		continue;
	
	void* library = dlopen("/usr/lib/libpcre.dylib", RTLD_LAZY);
	int (*pcre_config)(int, void *) = (int (*)(int, void *))dlsym(library, "pcre_config");
	if (pcre_config) {
		int value;
		pcre_config(PCRE_CONFIG_UTF8, &value);
	}	
        dlclose(library);
	
	return 0;
}
