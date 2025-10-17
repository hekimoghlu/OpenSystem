/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#ifndef PROFILING_H
#define PROFILING_H

#include "llvm/Analysis/ProfileDataTypes.h" /* for enum ProfilingType */

/* save_arguments - Save argc and argv as passed into the program for the file
 * we output.
 */
int save_arguments(int argc, const char **argv);

/*
 * Retrieves the file descriptor for the profile file.
 */
int getOutFile();

/* write_profiling_data - Write out a typed packet of profiling data to the
 * current output file.
 */
void write_profiling_data(enum ProfilingType PT, unsigned *Start,
                          unsigned NumElements);

#endif
