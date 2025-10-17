/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#include "util_cfgtree.h"
#include <stdlib.h>

ap_directive_t *ap_add_node(ap_directive_t **parent, ap_directive_t *current,
                            ap_directive_t *toadd, int child)
{
    if (current == NULL) {
        /* we just started a new parent */
        if (*parent != NULL) {
            (*parent)->first_child = toadd;
            toadd->parent = *parent;
        }
        if (child) {
            /* First item in config file or container is a container */
            *parent = toadd;
            return NULL;
        }
        return toadd;
    }
    current->next = toadd;
    toadd->parent = *parent;
    if (child) {
        /* switch parents, navigate into child */
        *parent = toadd;
        return NULL;
    }
    return toadd;
}


