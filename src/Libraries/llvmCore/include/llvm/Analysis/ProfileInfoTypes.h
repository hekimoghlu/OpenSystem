/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
#ifndef LLVM_ANALYSIS_PROFILEINFOTYPES_H
#define LLVM_ANALYSIS_PROFILEINFOTYPES_H

/* Included by libprofile. */
#if defined(__cplusplus)
extern "C" {
#endif

/* IDs to distinguish between those path counters stored in hashses vs arrays */
enum ProfilingStorageType {
  ProfilingArray = 1,
  ProfilingHash = 2
};

#include "llvm/Analysis/ProfileDataTypes.h"

/*
 * The header for tables that map path numbers to path counters.
 */
typedef struct {
  unsigned fnNumber; /* function number for these counters */
  unsigned numEntries;   /* number of entries stored */
} PathProfileHeader;

/*
 * Describes an entry in a tagged table for path counters.
 */
typedef struct {
  unsigned pathNumber;
  unsigned pathCounter;
} PathProfileTableEntry;

#if defined(__cplusplus)
}
#endif

#endif /* LLVM_ANALYSIS_PROFILEINFOTYPES_H */
