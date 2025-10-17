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

#ifndef RUBY_WIN32_DIR_H
#define RUBY_WIN32_DIR_H

#define DT_UNKNOWN 0
#define DT_DIR (S_IFDIR>>12)
#define DT_REG (S_IFREG>>12)
#define DT_LNK 10

struct direct
{
    long d_namlen;
    ino_t d_ino;
    char *d_name;
    char *d_altname; /* short name */
    short d_altlen;
    uint8_t d_type;
};
typedef struct {
    WCHAR *start;
    WCHAR *curr;
    long size;
    long nfiles;
    long loc;  /* [0, nfiles) */
    struct direct dirstr;
    char *bits;  /* used for d_isdir and d_isrep */
} DIR;


DIR*           rb_w32_opendir(const char*);
DIR*           rb_w32_uopendir(const char*);
struct direct* rb_w32_readdir(DIR *, rb_encoding *);
long           rb_w32_telldir(DIR *);
void           rb_w32_seekdir(DIR *, long);
void           rb_w32_rewinddir(DIR *);
void           rb_w32_closedir(DIR *);
char          *rb_w32_ugetcwd(char *, int);

#define opendir(s)   rb_w32_opendir((s))
#define readdir(d)   rb_w32_readdir((d), 0)
#define telldir(d)   rb_w32_telldir((d))
#define seekdir(d, l)   rb_w32_seekdir((d), (l))
#define rewinddir(d) rb_w32_rewinddir((d))
#define closedir(d)  rb_w32_closedir((d))

#endif /* RUBY_WIN32_DIR_H */
