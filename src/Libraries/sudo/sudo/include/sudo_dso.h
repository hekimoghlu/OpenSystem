/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#ifndef SUDO_DSO_H
#define SUDO_DSO_H

/* Values for sudo_dso_load() mode. */
#define SUDO_DSO_LAZY	 0x1
#define SUDO_DSO_NOW	 0x2
#define SUDO_DSO_GLOBAL	 0x4
#define SUDO_DSO_LOCAL	 0x8

/* Special handle arguments for sudo_dso_findsym(). */
#define SUDO_DSO_NEXT	 ((void *)-1)	/* Search subsequent objects. */
#define SUDO_DSO_DEFAULT ((void *)-2)	/* Use default search algorithm. */
#define SUDO_DSO_SELF	 ((void *)-3)	/* Search the caller itself. */

/* Internal structs for static linking of plugins. */
struct sudo_preload_symbol {
    const char *name;
    void *addr;
};
struct sudo_preload_table {
    const char *path;
    void *handle;
    struct sudo_preload_symbol *symbols;
};

/* Public functions. */
sudo_dso_public char *sudo_dso_strerror_v1(void);
sudo_dso_public int sudo_dso_unload_v1(void *handle);
sudo_dso_public void *sudo_dso_findsym_v1(void *handle, const char *symbol);
sudo_dso_public void *sudo_dso_load_v1(const char *path, int mode);
sudo_dso_public void sudo_dso_preload_table_v1(struct sudo_preload_table *table);

#define sudo_dso_strerror() sudo_dso_strerror_v1()
#define sudo_dso_unload(_a) sudo_dso_unload_v1((_a))
#define sudo_dso_findsym(_a, _b) sudo_dso_findsym_v1((_a), (_b))
#define sudo_dso_load(_a, _b) sudo_dso_load_v1((_a), (_b))
#define sudo_dso_preload_table(_a) sudo_dso_preload_table_v1((_a))

#endif /* SUDO_DSO_H */
