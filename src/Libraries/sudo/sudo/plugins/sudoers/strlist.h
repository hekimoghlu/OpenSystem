/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#ifndef SUDOERS_STRLIST_H
#define SUDOERS_STRLIST_H

/*
 * Simple string list with optional reference count.
 */
struct sudoers_string {
    STAILQ_ENTRY(sudoers_string) entries;
    char *str;
};
struct sudoers_str_list {
    struct sudoers_string *stqh_first;
    struct sudoers_string **stqh_last;
    unsigned int refcnt;
};

struct sudoers_str_list *str_list_alloc(void);
void str_list_free(void *v);
struct sudoers_string *sudoers_string_alloc(const char *s);
void sudoers_string_free(struct sudoers_string *ls);

#endif /* SUDOERS_STRLIST_H */
