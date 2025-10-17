/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#include "util.h"

#include "coretype.h"
#include "assocdat.h"


typedef struct assoc_data_item {
    const yasm_assoc_data_callback *callback;
    void *data;
} assoc_data_item;

struct yasm__assoc_data {
    assoc_data_item *vector;
    size_t size;
    size_t alloc;
};


yasm__assoc_data *
yasm__assoc_data_create(void)
{
    yasm__assoc_data *assoc_data = yasm_xmalloc(sizeof(yasm__assoc_data));

    assoc_data->size = 0;
    assoc_data->alloc = 2;
    assoc_data->vector = yasm_xmalloc(assoc_data->alloc *
                                      sizeof(assoc_data_item));

    return assoc_data;
}

void *
yasm__assoc_data_get(yasm__assoc_data *assoc_data,
                     const yasm_assoc_data_callback *callback)
{
    size_t i;

    if (!assoc_data)
        return NULL;

    for (i=0; i<assoc_data->size; i++) {
        if (assoc_data->vector[i].callback == callback)
            return assoc_data->vector[i].data;
    }
    return NULL;
}

yasm__assoc_data *
yasm__assoc_data_add(yasm__assoc_data *assoc_data_arg,
                     const yasm_assoc_data_callback *callback, void *data)
{
    yasm__assoc_data *assoc_data;
    assoc_data_item *item = NULL;
    size_t i;

    /* Create a new assoc_data if necessary */
    if (assoc_data_arg)
        assoc_data = assoc_data_arg;
    else
        assoc_data = yasm__assoc_data_create();

    /* See if there's already assocated data for this callback */
    for (i=0; i<assoc_data->size; i++) {
        if (assoc_data->vector[i].callback == callback) {
            item = &assoc_data->vector[i];
            break;
        }
    }

    /* No?  Then append a new one */
    if (!item) {
        assoc_data->size++;
        if (assoc_data->size > assoc_data->alloc) {
            assoc_data->alloc *= 2;
            assoc_data->vector =
                yasm_xrealloc(assoc_data->vector,
                              assoc_data->alloc * sizeof(assoc_data_item));
        }
        item = &assoc_data->vector[assoc_data->size-1];
        item->callback = callback;
        item->data = NULL;
    }

    /* Delete existing data (if any) */
    if (item->data && item->data != data)
        item->callback->destroy(item->data);

    item->data = data;

    return assoc_data;
}

void
yasm__assoc_data_destroy(yasm__assoc_data *assoc_data)
{
    size_t i;

    if (!assoc_data)
        return;

    for (i=0; i<assoc_data->size; i++)
        assoc_data->vector[i].callback->destroy(assoc_data->vector[i].data);
    yasm_xfree(assoc_data->vector);
    yasm_xfree(assoc_data);
}

void
yasm__assoc_data_print(const yasm__assoc_data *assoc_data, FILE *f,
                       int indent_level)
{
    /*TODO*/
}
