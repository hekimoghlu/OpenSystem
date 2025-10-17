/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#include<config.h>

#include <stdlib.h>
#include <io.h>
#include <string.h>
#include <errno.h>
#include "dirent.h"

#ifndef _WIN32
#error Only implemented for Win32
#endif

struct _dirent_dirinfo {
    int             magic;
    long            n_entries;
    long            nc_entries;
    long            cursor;
    struct dirent **entries;
};
#define DIRINFO_MAGIC 0xf8c0639d
#define IS_DP(p) ((p) && ((DIR *)(p))->magic == DIRINFO_MAGIC)

#define INITIAL_ENTRIES 16

/**
 * Create a filespec for use with _findfirst() using a path spec
 *
 * If the last component of the path spec contains wildcards, we let
 * it be.  If the last component doesn't end with a slash, we add one.
 */
static const char *
filespec_from_dir_path(const char * path, char * buffer, size_t cch_buffer)
{
    char *comp, *t;
    size_t pos;
    int found_sep = 0;

    if (strcpy_s(buffer, cch_buffer, path) != 0)
        return NULL;

    comp = strrchr(buffer, '\\');
    if (comp == NULL)
        comp = buffer;
    else
        found_sep = 1;

    t = strrchr(comp, '/');
    if (t != NULL) {
        comp = t;
        found_sep = 1;
    }

    if (found_sep)
        comp++;

    pos = strcspn(comp, "*?");
    if (comp[pos] != '\0')
        return buffer;

    /* We don't append a slash if pos == 0 because that changes the
     * meaning:
     *
     * "*.*" is all files in the current directory.
     * "\*.*" is all files in the root directory of the current drive.
     */
    if (pos > 0 && comp[pos - 1] != '\\' &&
        comp[pos - 1] != '/') {
        strcat_s(comp, cch_buffer - (comp - buffer), "\\");
    }

    strcat_s(comp, cch_buffer - (comp - buffer), "*.*");

    return buffer;
}

ROKEN_LIB_FUNCTION DIR * ROKEN_LIB_CALL
opendir(const char * path)
{
    DIR *              dp;
    struct _finddata_t fd;
    intptr_t           fd_handle;
    const char         *filespec;
    char               path_buffer[1024];

    memset(&fd, 0, sizeof(fd));

    filespec = filespec_from_dir_path(path, path_buffer, sizeof(path_buffer)/sizeof(char));
    if (filespec == NULL)
        return NULL;

    fd_handle = _findfirst(filespec, &fd);

    if (fd_handle == -1)
        return NULL;

    dp = malloc(sizeof(*dp));
    if (dp == NULL)
        goto done;

    memset(dp, 0, sizeof(*dp));
    dp->magic      = DIRINFO_MAGIC;
    dp->cursor     = 0;
    dp->n_entries  = 0;
    dp->nc_entries = INITIAL_ENTRIES;
    dp->entries    = calloc(dp->nc_entries, sizeof(dp->entries[0]));

    if (dp->entries == NULL) {
        closedir(dp);
        dp = NULL;
        goto done;
    }

    do {
        size_t len = strlen(fd.name);
        struct dirent * e;

        if (dp->n_entries == dp->nc_entries) {
	    struct dirent ** ne;

            dp->nc_entries *= 2;
            ne = realloc(dp->entries, sizeof(dp->entries[0]) * dp->nc_entries);

            if (ne == NULL) {
                closedir(dp);
                dp = NULL;
                goto done;
            }

	    dp->entries = ne;
        }

        e = malloc(sizeof(*e) + len * sizeof(char));
        if (e == NULL) {
            closedir(dp);
            dp = NULL;
            goto done;
        }

        e->d_ino = 0;           /* no inodes :( */
        strcpy_s(e->d_name, len + 1, fd.name);

        dp->entries[dp->n_entries++] = e;

    } while (_findnext(fd_handle, &fd) == 0);

 done:
    if (fd_handle != -1)
        _findclose(fd_handle);

    return dp;
}

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
closedir(DIR * dp)
{
    if (!IS_DP(dp))
        return EINVAL;

    if (dp->entries) {
        long i;

        for (i=0; i < dp->n_entries; i++) {
            free(dp->entries[i]);
        }

        free(dp->entries);
    }

    free(dp);

    return 0;
}

ROKEN_LIB_FUNCTION struct dirent * ROKEN_LIB_CALL
readdir(DIR * dp)
{
    if (!IS_DP(dp) ||
        dp->cursor < 0 ||
        dp->cursor >= dp->n_entries)

        return NULL;

    return dp->entries[dp->cursor++];
}

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rewinddir(DIR * dp)
{
    if (IS_DP(dp))
        dp->cursor = 0;
}

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
seekdir(DIR * dp, long offset)
{
    if (IS_DP(dp) && offset >= 0 && offset < dp->n_entries)
        dp->cursor = offset;
}

ROKEN_LIB_FUNCTION long ROKEN_LIB_CALL
telldir(DIR * dp)
{
    return dp->cursor;
}
