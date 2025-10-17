/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <locale.h>

void main(int argc, char **argv) {

	HINSTANCE hinst;
	WCHAR buffer[128];
	unsigned char winbuf[128],oembuf[128];
	unsigned int number;

	if (argc <3)
		return;

   	hinst = LoadLibrary(argv[1]);

	number = atoi(argv[2]);
	printf("Load String returns %i\n",	
		LoadStringW(hinst, number, buffer, sizeof(buffer)));

	WideCharToMultiByte(CP_OEMCP,
						0,
						buffer,
						-1,
						winbuf,
						128,
						NULL,
						NULL);

	CharToOem(winbuf,oembuf);
	printf("oem: %s\n",oembuf);
}
