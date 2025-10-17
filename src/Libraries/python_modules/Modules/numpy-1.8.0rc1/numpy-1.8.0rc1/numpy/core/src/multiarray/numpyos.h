/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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

#ifndef _NPY_NUMPYOS_H_
#define _NPY_NUMPYOS_H_

NPY_NO_EXPORT char*
NumPyOS_ascii_formatd(char *buffer, size_t buf_size,
                      const char *format,
                      double val, int decimal);

NPY_NO_EXPORT char*
NumPyOS_ascii_formatf(char *buffer, size_t buf_size,
                      const char *format,
                      float val, int decimal);

NPY_NO_EXPORT char*
NumPyOS_ascii_formatl(char *buffer, size_t buf_size,
                      const char *format,
                      long double val, int decimal);

NPY_NO_EXPORT double
NumPyOS_ascii_strtod(const char *s, char** endptr);

NPY_NO_EXPORT int
NumPyOS_ascii_ftolf(FILE *fp, double *value);

NPY_NO_EXPORT int
NumPyOS_ascii_isspace(char c);

#endif
