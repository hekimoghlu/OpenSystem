/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#ifndef SIEVE_H
#define SIEVE_H

#ifndef U_LOTS_OF_TIMES
#define U_LOTS_OF_TIMES 1000000
#endif

#include "unicode/utypes.h"
/**
 * Calculate the standardized sieve time (1 run)
 */
U_CAPI double uprv_calcSieveTime(void);

/**
 * Calculate the mean time, with margin of error
 * @param times array of times (modified/sorted)
 * @param timeCount length of array - on return, how many remain after throwing out outliers
 * @param marginOfError out parameter: gives +/- margin of err at 95% confidence
 * @return the mean time, or negative if error/imprecision.
 */
U_CAPI double uprv_getMeanTime(double *times, uint32_t *timeCount, double *marginOfError);

/**
 * Get the standardized sieve time. (Doesn't recalculate if already computed.
 * @param marginOfError out parameter: gives +/- margin of error at 95% confidence.
 * @return the mean time, or negative if error/imprecision.
 */
U_CAPI double uprv_getSieveTime(double *marginOfError);

#endif
