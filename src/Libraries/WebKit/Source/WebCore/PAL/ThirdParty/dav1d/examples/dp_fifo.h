/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
/*
 * Dav1dPlay FIFO helper
 */

typedef struct dp_fifo Dav1dPlayPtrFifo;

/* Create a FIFO
 *
 * Creates a FIFO with the given capacity.
 * If the capacity is reached, new inserts into the FIFO
 * will block until enough space is available again.
 */
Dav1dPlayPtrFifo *dp_fifo_create(size_t capacity);

/* Destroy a FIFO
 *
 * The FIFO must be empty before it is destroyed!
 */
void dp_fifo_destroy(Dav1dPlayPtrFifo *fifo);

/* Shift FIFO
 *
 * Return the first item from the FIFO, thereby removing it from
 * the FIFO and making room for new entries.
 */
void *dp_fifo_shift(Dav1dPlayPtrFifo *fifo);

/* Push to FIFO
 *
 * Add an item to the end of the FIFO.
 * If the FIFO is full, this call will block until there is again enough
 * space in the FIFO, so calling this from the "consumer" thread if no
 * other thread will call dp_fifo_shift will lead to a deadlock.
 */
void dp_fifo_push(Dav1dPlayPtrFifo *fifo, void *element);

void dp_fifo_flush(Dav1dPlayPtrFifo *fifo, void (*destroy_elem)(void *));
