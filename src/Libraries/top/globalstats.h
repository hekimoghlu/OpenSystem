/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#ifndef GLOBALSTATS_H
#define GLOBALSTATS_H

#include <curses.h>
#include <stdbool.h>

void *top_globalstats_create(WINDOW *parent);
void top_globalstats_draw(void *ptr);
bool top_globalstats_update(void *ptr, const void *sample);
bool top_globalstats_resize(void *ptr, int width, int height, int *consumed_height);
void top_globalstats_iterate(void *ptr, bool (*iter)(char *, void *), void *dataptr);

/* This resets the maximum width of the windows, typically after a relayout. */
void top_globalstats_reset(void *ptr);

#endif
