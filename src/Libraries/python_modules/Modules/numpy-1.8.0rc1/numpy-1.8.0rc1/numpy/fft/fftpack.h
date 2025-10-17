/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#ifdef __cplusplus
extern "C" {
#endif

#define DOUBLE

#ifdef DOUBLE
#define Treal double
#else
#define Treal float
#endif

extern void cfftf(int N, Treal data[], const Treal wrk[]);
extern void cfftb(int N, Treal data[], const Treal wrk[]);
extern void cffti(int N, Treal wrk[]);

extern void rfftf(int N, Treal data[], const Treal wrk[]);
extern void rfftb(int N, Treal data[], const Treal wrk[]);
extern void rffti(int N, Treal wrk[]);

#ifdef __cplusplus
}
#endif
