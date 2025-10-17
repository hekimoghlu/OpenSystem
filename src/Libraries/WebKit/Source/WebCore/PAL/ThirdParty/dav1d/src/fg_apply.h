/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#ifndef DAV1D_SRC_FG_APPLY_H
#define DAV1D_SRC_FG_APPLY_H

#include "dav1d/picture.h"

#include "common/bitdepth.h"

#include "src/filmgrain.h"

#ifdef BITDEPTH
# define array_decl(type, name, sz) type name sz
#else
# define array_decl(type, name, sz) void *name
#endif

bitfn_decls(void dav1d_apply_grain,
            const Dav1dFilmGrainDSPContext *const dsp,
            Dav1dPicture *const out, const Dav1dPicture *const in);
bitfn_decls(void dav1d_prep_grain,
            const Dav1dFilmGrainDSPContext *const dsp,
            Dav1dPicture *const out, const Dav1dPicture *const in,
            array_decl(uint8_t, scaling, [3][SCALING_SIZE]),
            array_decl(entry, grain_lut, [3][GRAIN_HEIGHT+1][GRAIN_WIDTH]));
bitfn_decls(void dav1d_apply_grain_row,
            const Dav1dFilmGrainDSPContext *const dsp,
            Dav1dPicture *const out, const Dav1dPicture *const in,
            array_decl(const uint8_t, scaling, [3][SCALING_SIZE]),
            array_decl(const entry, grain_lut, [3][GRAIN_HEIGHT+1][GRAIN_WIDTH]),
            const int row);

#endif /* DAV1D_SRC_FG_APPLY_H */
