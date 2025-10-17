/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 18, 2022.
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
/*!\file
 * \brief Describes film grain synthesis
 *
 */
#ifndef AOM_AV1_DECODER_GRAIN_SYNTHESIS_H_
#define AOM_AV1_DECODER_GRAIN_SYNTHESIS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "aom_dsp/grain_params.h"
#include "aom/aom_image.h"

/*!\brief Add film grain
 *
 * Add film grain to an image
 *
 * Returns 0 for success, -1 for failure
 *
 * \param[in]    grain_params     Grain parameters
 * \param[in]    src              Source image
 * \param[out]   dst              Resulting image with grain
 */
int av1_add_film_grain(const aom_film_grain_t *grain_params,
                       const aom_image_t *src, aom_image_t *dst);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_DECODER_GRAIN_SYNTHESIS_H_
