/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#ifndef YASM_ASSOCDAT_H
#define YASM_ASSOCDAT_H

#ifndef YASM_LIB_DECL
#define YASM_LIB_DECL
#endif

/** Associated data container. */
typedef struct yasm__assoc_data yasm__assoc_data;

/** Create an associated data container. */
YASM_LIB_DECL
/*@only@*/ yasm__assoc_data *yasm__assoc_data_create(void);

/** Get associated data for a data callback.
 * \param assoc_data    container of associated data
 * \param callback      callback used when adding data
 * \return Associated data (NULL if none).
 */
YASM_LIB_DECL
/*@dependent@*/ /*@null@*/ void *yasm__assoc_data_get
    (/*@null@*/ yasm__assoc_data *assoc_data,
     const yasm_assoc_data_callback *callback);

/** Add associated data to a associated data container.
 * \attention Deletes any existing associated data for that data callback.
 * \param assoc_data    container of associated data
 * \param callback      callback
 * \param data          data to associate
 */
YASM_LIB_DECL
/*@only@*/ yasm__assoc_data *yasm__assoc_data_add
    (/*@null@*/ /*@only@*/ yasm__assoc_data *assoc_data,
     const yasm_assoc_data_callback *callback,
     /*@only@*/ /*@null@*/ void *data);

/** Destroy all associated data in a container. */
YASM_LIB_DECL
void yasm__assoc_data_destroy
    (/*@null@*/ /*@only@*/ yasm__assoc_data *assoc_data);

/** Print all associated data in a container. */
YASM_LIB_DECL
void yasm__assoc_data_print(const yasm__assoc_data *assoc_data, FILE *f,
                            int indent_level);

#endif
