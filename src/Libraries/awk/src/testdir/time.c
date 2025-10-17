/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 5, 2022.
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
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/times.h>
#include <time.h>

int main(int argc, char *argv[])
{
	struct tms before, after;
	char cmd[10000];
	int i;
	double fudge = 100.0;	/* should be CLOCKS_PER_SEC but that gives nonsense */

	times(&before);

	/* ... place code to be timed here ... */
	cmd[0] = 0;
	for (i = 1; i < argc; i++)
		sprintf(cmd+strlen(cmd), "%s ", argv[i]);
	sprintf(cmd+strlen(cmd), "\n");
	/* printf("cmd = [%s]\n", cmd); */
	system(cmd);

	times(&after);

	fprintf(stderr, "user %6.3f\n", (after.tms_cutime - before.tms_cutime)/fudge);
	fprintf(stderr, "sys  %6.3f\n", (after.tms_cstime - before.tms_cstime)/fudge);

	return 0;
}
