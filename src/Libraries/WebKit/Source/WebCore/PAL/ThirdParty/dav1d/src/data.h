/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 29, 2022.
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
#ifndef DAV1D_SRC_DATA_H
#define DAV1D_SRC_DATA_H

#include "dav1d/data.h"

void dav1d_data_ref(Dav1dData *dst, const Dav1dData *src);

/**
 * Copy the source properties to the destination and increase the
 * user_data's reference count (if it's not NULL).
 */
void dav1d_data_props_copy(Dav1dDataProps *dst, const Dav1dDataProps *src);

void dav1d_data_props_set_defaults(Dav1dDataProps *props);

uint8_t *dav1d_data_create_internal(Dav1dData *buf, size_t sz);
int dav1d_data_wrap_internal(Dav1dData *buf, const uint8_t *ptr, size_t sz,
                             void (*free_callback)(const uint8_t *data,
                                                   void *user_data),
                             void *user_data);
int dav1d_data_wrap_user_data_internal(Dav1dData *buf,
                                       const uint8_t *user_data,
                                       void (*free_callback)(const uint8_t *user_data,
                                                             void *cookie),
                                       void *cookie);
void dav1d_data_unref_internal(Dav1dData *buf);
void dav1d_data_props_unref_internal(Dav1dDataProps *props);

#endif /* DAV1D_SRC_DATA_H */
