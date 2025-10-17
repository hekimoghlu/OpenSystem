/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#ifndef SUDOERS_NSS_H
#define SUDOERS_NSS_H

struct passwd;
struct userspec_list;
struct defaults_list;

/* XXX - parse_tree, ret_if_found and ret_if_notfound should be private */
struct sudo_nss {
    TAILQ_ENTRY(sudo_nss) entries;
    const char *source;
    int (*open)(struct sudo_nss *nss);
    int (*close)(struct sudo_nss *nss);
    struct sudoers_parse_tree *(*parse)(struct sudo_nss *nss);
    int (*query)(struct sudo_nss *nss, struct passwd *pw);
    int (*getdefs)(struct sudo_nss *nss);
    void *handle;
    struct sudoers_parse_tree *parse_tree;
    bool ret_if_found;
    bool ret_if_notfound;
};

TAILQ_HEAD(sudo_nss_list, sudo_nss);

struct sudo_nss_list *sudo_read_nss(void);
bool sudo_nss_can_continue(struct sudo_nss *nss, int match);

#endif /* SUDOERS_NSS_H */
