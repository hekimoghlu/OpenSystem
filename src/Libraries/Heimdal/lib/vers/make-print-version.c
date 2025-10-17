/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 3, 2025.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <string.h>

#ifdef KRB5
extern const char *heimdal_version;
#endif
#include <version.h>

int
main(int argc, char **argv)
{
    FILE *f;
    if(argc != 2)
	return 1;
    if (strcmp(argv[1], "--version") == 0) {
	printf("some version");
	return 0;
    }
    f = fopen(argv[1], "w");
    if(f == NULL)
	return 1;
    fprintf(f, "#define VERSIONLIST \"");
#ifdef KRB5
    fprintf(f, "%s", heimdal_version);
#endif
    fprintf(f, "\"\n");
    fclose(f);
    return 0;
}
