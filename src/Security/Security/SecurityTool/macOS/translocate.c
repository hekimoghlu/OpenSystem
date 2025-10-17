/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include <dirent.h>

#include <CoreFoundation/CoreFoundation.h>

#include <Security/SecTranslocate.h>

#include "security_tool.h"
#include "translocate.h"

static CFURLRef CFURLfromPath(const char * path, Boolean isDir)
{
    return CFURLCreateFromFileSystemRepresentation(NULL, (UInt8*)path, strlen(path), isDir);
}

static char * PathFromCFURL(CFURLRef url)
{
    char* path = malloc(PATH_MAX);

    if (!path)
    {
        goto done;
    }

    if (!CFURLGetFileSystemRepresentation(url, true, (UInt8*)path, PATH_MAX))
    {
        free(path);
        path = NULL;
    }

done:
    return path;
}

static Boolean PathIsDir(const char * path)
{
    Boolean result = false;

    if(!path)
    {
        goto done;
    }

    DIR* d = opendir(path);

    if(d)
    {
        result = true;
        closedir(d);
    }

done:
    return result;
}

static void SafeCFRelease(CFTypeRef ref)
{
    if (ref)
    {
        CFRelease(ref);
    }
}

/* return 2 = bad args, anything else is ignored */

int translocate_policy(int argc, char * const *argv)
{
    int result = -1;

    if (argc != 2)
    {
        return SHOW_USAGE_MESSAGE;
    }

    CFURLRef inUrl = CFURLfromPath(argv[1], PathIsDir(argv[1]));
    bool should = false;
    CFErrorRef error = NULL;

    if(!inUrl)
    {
        printf("Error: failed to create url for: %s\n", argv[1]);
        goto done;
    }

    if (!SecTranslocateURLShouldRunTranslocated(inUrl, &should, &error))
    {
        int err = (int)CFErrorGetCode(error);
        printf("Error: failed while trying to check policy for %s (errno: %d, %s)\n", argv[1], err, strerror(err));
        goto done;
    }

    printf("\t%s\n", should ? "Would translocate": "Would not translocate");

    result = 0;

done:
    SafeCFRelease(inUrl);
    SafeCFRelease(error);

    return result;
}

int translocate_check(int argc, char * const *argv)
{
    int result = -1;

    if (argc != 2)
    {
        return SHOW_USAGE_MESSAGE;
    }

    CFURLRef inUrl = CFURLfromPath(argv[1], PathIsDir(argv[1]));
    bool is = false;
    CFErrorRef error = NULL;

    if(!inUrl)
    {
        printf("Error: failed to create url for: %s\n", argv[1]);
        goto done;
    }

    if (!SecTranslocateIsTranslocatedURL(inUrl, &is, &error))
    {
        int err = (int)CFErrorGetCode(error);
        printf("Error: failed while trying to check status for %s (errno: %d, %s)\n", argv[1], err, strerror(err));
        goto done;
    }

    printf("\t%s\n", is ? "TRANSLOCATED": "NOT TRANSLOCATED");

    result = 0;

done:
    SafeCFRelease(inUrl);
    SafeCFRelease(error);

    return result;
}

int translocate_original_path(int argc, char * const * argv)
{
    int result = -1;

    if (argc != 2)
    {
        return SHOW_USAGE_MESSAGE;
    }

    CFURLRef inUrl = CFURLfromPath(argv[1], PathIsDir(argv[1]));
    CFURLRef outUrl = NULL;
    CFErrorRef error = NULL;
    char* outPath = NULL;

    if(!inUrl)
    {
        printf("Error: failed to create url for: %s\n", argv[1]);
        goto done;
    }

    outUrl = SecTranslocateCreateOriginalPathForURL(inUrl, &error);

    if (!outUrl)
    {
        int err = (int)CFErrorGetCode(error);
        printf("Error: failed while trying to find original path for %s (errno: %d, %s)\n", argv[1], err, strerror(err));
        goto done;
    }

    outPath = PathFromCFURL(outUrl);

    if( !outPath )
    {
        printf("Error: failed to convert out url to string for %s\n", argv[1]);
        goto done;
    }

    printf("Original Path: (note if this is what you passed in then that path is not translocated)\n\t%s\n",outPath);

    free(outPath);
    result = 0;

done:
    SafeCFRelease(inUrl);
    SafeCFRelease(outUrl);
    SafeCFRelease(error);

    return result;
}

