/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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

#include <stdio.h>

/* This is a cc optimiser bug for ultrix 4.3, mips CPU.
 * What happens is that the compiler, due to the (a)&7,
 * does
 * i=a&7;
 * i--;
 * i*=4;
 * Then uses i as the offset into a jump table.
 * The problem is that a value of 0 generates an offset of
 * 0xfffffffc.
 */

main()
	{
	f(5);
	f(0);
	}

int f(a)
int a;
	{
	switch(a&7)
		{
	case 7:
		printf("7\n");
	case 6:
		printf("6\n");
	case 5:
		printf("5\n");
	case 4:
		printf("4\n");
	case 3:
		printf("3\n");
	case 2:
		printf("2\n");
	case 1:
		printf("1\n");
#ifdef FIX_BUG
	case 0:
		;
#endif
		}
	}	

