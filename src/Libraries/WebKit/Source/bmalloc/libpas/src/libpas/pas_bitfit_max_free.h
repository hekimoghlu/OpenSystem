/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
#ifndef PAS_BITFIT_MAX_FREE_H
#define PAS_BITFIT_MAX_FREE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

typedef uint8_t pas_bitfit_max_free;

#define PAS_BITFIT_MAX_FREE_MAX_VALID   ((pas_bitfit_max_free)253)
#define PAS_BITFIT_MAX_FREE_UNPROCESSED ((pas_bitfit_max_free)254)
#define PAS_BITFIT_MAX_FREE_EMPTY       ((pas_bitfit_max_free)255)

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_MAX_FREE_H */

