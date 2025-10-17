/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
#ifndef AOM_AOM_DSP_BINARY_CODES_WRITER_H_
#define AOM_AOM_DSP_BINARY_CODES_WRITER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include "config/aom_config.h"

#include "aom/aom_integer.h"
#include "aom_dsp/bitwriter.h"
#include "aom_dsp/bitwriter_buffer.h"

// Finite subexponential code that codes a symbol v in [0, n-1] with parameter k
// based on a reference ref also in [0, n-1].
void aom_write_primitive_refsubexpfin(aom_writer *w, uint16_t n, uint16_t k,
                                      uint16_t ref, uint16_t v);

// Functions that counts bits for the above primitives
int aom_count_primitive_refsubexpfin(uint16_t n, uint16_t k, uint16_t ref,
                                     uint16_t v);
int aom_count_signed_primitive_refsubexpfin(uint16_t n, uint16_t k, int16_t ref,
                                            int16_t v);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AOM_DSP_BINARY_CODES_WRITER_H_
